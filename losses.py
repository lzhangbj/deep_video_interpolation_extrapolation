import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from math import exp
from utils.net_utils import *
from nets.vgg import *
from collections import OrderedDict

import torchvision

#####################################################################################################
############################################ ssim loss ###############################################
#####################################################################################################
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return -_ssim(img1, img2, window, self.window_size, channel, self.size_average) + 1

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)


#####################################################################################################
################################################ psnr loss ##########################################
#####################################################################################################
class PSNR(torch.nn.Module):
    def __init__(self, max_level=1):
        super(PSNR, self).__init__()
        self.max_level = max_level

    def forward(self, pred, gt):
        assert (pred.size() == gt.size())
        _,_,h,w = pred.size()
        psnr = 0
        for i in range(pred.size(0)):
            delta = (pred[i, :, :, :] - gt[i, :, :, :])
            delta = torch.mean(torch.pow(delta, 2))
            psnr += 10 * torch.log10(self.max_level * self.max_level / delta)
        return psnr/pred.size(0)


#####################################################################################################
################################################ iou  loss ##########################################
#####################################################################################################
class IoU(torch.nn.Module):
    def __init__(self):
        super(IoU, self).__init__()

    def forward(self, pred, gt):
        assert (pred.size() == gt.size())
        bs,h,w = gt.size()
        true_pixs = (pred == gt).float()
        iou = torch.sum(true_pixs) / (bs*h*w)
        return iou


#####################################################################################################
################################################# gdl loss ##########################################
#####################################################################################################
class GDLLoss(torch.nn.Module):
    def __init__(self):
        super(GDLLoss, self).__init__()

    def forward(self, input, gt):
        bs, c, h, w = input.size()

        w_gdl = torch.abs(input[:,:,:,1:] - input[:,:,:,:w-1])
        h_gdl = torch.abs(input[:,:,1:,:] - input[:,:,:h-1,:])

        gt_w_gdl = torch.abs(gt[:,:,:,1:] - gt[:,:,:,:w-1])
        gt_h_gdl = torch.abs(gt[:,:,1:,:] - gt[:,:,:h-1,:])
        
        loss = torch.mean(torch.abs(w_gdl-gt_w_gdl)) + torch.mean(torch.abs(h_gdl-gt_h_gdl))
        return loss


#####################################################################################################
################################################ vgg loss ###########################################
#####################################################################################################
class VGGLoss(torch.nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        vgg19 = torchvision.models.vgg19(pretrained=True)
        self.vgg_net = my_vgg(vgg19)
        for param in self.vgg_net.parameters():
            param.requires_grad = False


    def forward(self, input, gt, normed=True):
        '''
            if normed, input is already normed by mean and std
            otherwise, it is in the range(-1, 1)
        '''
        if not normed:
            input = preprocess_norm(input, input.is_cuda)
            gt = preprocess_norm(gt, gt.is_cuda)
        input_feature = self.vgg_net(input)
        gt_feature = self.vgg_net(gt)
        loss = 0
        for i in range(len(input_feature)):
            loss += (input_feature[i] - gt_feature[i]).abs().mean()
        return loss/len(input_feature)

#####################################################################################################
################################################ rgb loss ###########################################
#####################################################################################################
class RGBLoss(torch.nn.Module):
    def __init__(self, args, window_size = 11, size_average = True):
        super(RGBLoss, self).__init__()  
        self.vgg_loss = VGGLoss()
        self.gdl_loss = GDLLoss()
        self.ssim_loss = SSIM(window_size, size_average)
        self.l1_loss = torch.nn.L1Loss()
        self.args = args

    def forward(self, input, gt, normed=True, length=1):
        vgg_loss = 0
        ssim_loss = 0
        for i in range(length):
            vgg_loss += self.vgg_loss(input[:,3*i:3*(i+1)], gt[:, 3*i:3*(i+1)], normed)
            ssim_loss+=self.ssim_loss(input[:,3*i:3*(i+1)], gt[:, 3*i:3*(i+1)])
        vgg_loss/=length
        ssim_loss/=length
        # return self.args.l1_weight*self.l1_loss(input, gt), \
        #         self.args.gdl_weight*self.gdl_loss(input, gt), \
        #         self.args.vgg_weight*vgg_loss, \
        #         self.args.ssim_weight*ssim_loss
        return  OrderedDict( 
                [('l1_loss', self.args.l1_weight*self.l1_loss(input, gt)), 
                ('gdl_loss', self.args.gdl_weight*self.gdl_loss(input, gt)), 
                ('vgg_loss', self.args.vgg_weight*vgg_loss), 
                ('ssim_loss',self.args.ssim_weight*ssim_loss)]
                )


#####################################################################################################
################################################ gan loss ###########################################
#####################################################################################################
class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.LongTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        device_id = input.get_device()
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False).cuda(device_id)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False).cuda(device_id)
            target_tensor = self.fake_label_var
        return target_tensor

    def forward(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)

class GANMapLoss(nn.Module):
    def __init__(self):
        super(GANMapLoss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, input, label_map, target_is_real, gen=False):
        loss = 0
        bs, c, h, w = label_map.size()

        if not gen:
            fake_num = torch.sum(1-label_map)
            real_num = 2*bs*h*w - fake_num
            real_ratio = float(fake_num.cpu().item()) / (2*bs*h*w)
            fake_ratio = 1-real_ratio
        else:
            real_ratio = 1

        real_ratio = 1
        fake_ratio = 1
        for input_i in input:
            pred = input_i[-1]
            if target_is_real:
                loss += real_ratio*self.loss(pred, pred.new(pred.size()).fill_(1))
            else:
                label_map_temp = label_map
                label_map_temp.fill_(0)
                # label_map_temp = F.interpolate(label_map, pred.size()[2:], mode='nearest')
                # f2r_ratio = fake_num/(2*h*w - fake_num)
                fake_loss = torch.mean(torch.abs(pred-label_map_temp)*(1-label_map_temp)) * fake_ratio
                real_loss = torch.mean(torch.abs(pred-label_map_temp)*label_map_temp) * real_ratio
                
                loss += (fake_loss + real_loss)
                # loss += self.loss(pred, label_map_temp)
        return loss      