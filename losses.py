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
############################################ ssim loss ##############################################
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

        w_gdl = input[:,:,:,1:] - input[:,:,:,:w-1]
        h_gdl = input[:,:,1:,:] - input[:,:,:h-1,:]

        gt_w_gdl = gt[:,:,:,1:] - gt[:,:,:,:w-1]
        gt_h_gdl = gt[:,:,1:,:] - gt[:,:,:h-1,:]
        
        loss = torch.mean(torch.abs(w_gdl-gt_w_gdl)) + torch.mean(torch.abs(h_gdl-gt_h_gdl))
        return loss/2


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
        self.Loss = nn.MSELoss()


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
            loss += self.Loss(input_feature[i], gt_feature[i])
        return loss/len(input_feature)

class VGGCosineLoss(torch.nn.Module):
    def __init__(self):
        super(VGGCosineLoss, self).__init__()
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
        score = 0
        for i in range(len(input_feature)):
            input_feature_ = input_feature[i] / torch.sqrt(torch.sum(input_feature[i]**2, dim=1, keepdim=True))
            gt_feature_ = gt_feature[i] / torch.sqrt(torch.sum(gt_feature[i]**2, dim=1, keepdim=True))
            score += torch.mean(torch.sum(gt_feature_*input_feature_, dim=1))
        score/=len(input_feature)
        return score


#####################################################################################################
################################################ rgb loss ###########################################
#####################################################################################################
class RGBLoss(torch.nn.Module):
    def __init__(self, args, window_size = 11, size_average = True, sharp=False, refine=False):
        super(RGBLoss, self).__init__()  
        self.vgg_loss = VGGLoss()
        self.gdl_loss = GDLLoss()
        self.refine=refine
        self.ssim_loss = SSIM(window_size, size_average)
        self.l1_loss = torch.nn.L1Loss()
        if sharp:
            self.sharp_loss = SharpenessLoss()
        self.args = args

    def forward(self, input, gt, normed=True, length=1, refine_scale=1, step=1, res=1):
        vgg_loss = 0
        ssim_loss = 0
        for i in range(length):
            # if res==1:
            vgg_loss += self.vgg_loss(input[:,3*i:3*(i+1)], gt[:, 3*i:3*(i+1)], normed)
            ssim_loss+=self.ssim_loss(input[:,3*i:3*(i+1)], gt[:, 3*i:3*(i+1)])
        # if res==1:
        vgg_loss/=length
        ssim_loss/=length
        # return self.args.l1_weight*self.l1_loss(input, gt), \
        #         self.args.gdl_weight*self.gdl_loss(input, gt), \
        #         self.args.vgg_weight*vgg_loss, \
        #         self.args.ssim_weight*ssim_loss
        if not self.refine:
            return  OrderedDict( 
                    [('l1_loss', self.args.l1_weight*self.l1_loss(input, gt)), 
                    ('gdl_loss', self.args.gdl_weight*self.gdl_loss(input, gt)), 
                    ('vgg_loss', self.args.vgg_weight*vgg_loss), 
                    ('ssim_loss',self.args.ssim_weight*ssim_loss)]
                    # ('sharp_loss',self.args.sharp_weight*self.sharp_loss(input, gt))]
                    )
        else:
            if res == 1:
                if refine_scale==1 and step<1000 and self.args.n_scales!=1:
                    return  OrderedDict( 
                            [('refine_{:.2f}_l1_loss'.format(refine_scale), 0*self.l1_loss(input, gt)), 
                            ('refine_{:.2f}_gdl_loss'.format(refine_scale), 0*self.gdl_loss(input, gt)), 
                            ('refine_{:.2f}_vgg_loss'.format(refine_scale), 0*vgg_loss)]
                            )              
                else:
                    return  OrderedDict( 
                            [('refine_{:.2f}_l1_loss'.format(refine_scale), self.args.refine_l1_weight*self.l1_loss(input, gt)), 
                            ('refine_{:.2f}_gdl_loss'.format(refine_scale), self.args.refine_gdl_weight*self.gdl_loss(input, gt)), 
                            ('refine_{:.2f}_vgg_loss'.format(refine_scale), self.args.refine_vgg_weight*vgg_loss)]
                            )
            else:
                return  OrderedDict( 
                            [('refine_{:d}_l1_loss'.format(res), self.args.refine_l1_weight*self.l1_loss(input, gt)), 
                            ('refine_{:d}_gdl_loss'.format(res), self.args.refine_gdl_weight*self.gdl_loss(input, gt)),
                            ('refine_{:d}_vgg_loss'.format(res), self.args.refine_vgg_weight*self.gdl_loss(input, gt))]
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
                label_map_temp = F.interpolate(label_map, size=list(pred.size()[2:]), mode='nearest')
                label_map_temp.fill_(0)
                # label_map_temp = F.interpolate(label_map, pred.size()[2:], mode='nearest')
                # f2r_ratio = fake_num/(2*h*w - fake_num)
                fake_loss = torch.mean(torch.abs(pred-label_map_temp)*(1-label_map_temp)) * fake_ratio
                real_loss = torch.mean(torch.abs(pred-label_map_temp)*label_map_temp) * real_ratio
                
                loss += (fake_loss + real_loss)
                # loss += self.loss(pred, label_map_temp)
        loss /= len(input)
        return loss      


class SharpenessLoss(nn.Module):
    def __init__(self):
        super(SharpenessLoss, self).__init__()
        self.loss = nn.L1Loss()
        self.pool = nn.MaxPool2d(5, padding=2)

    def forward(self, input, gt):
        input_max_pool = self.pool(input)
        input_min_pool = 1-self.pool(1-input)

        gt_max_pool = self.pool(gt)
        gt_min_pool = 1-self.pool(1-gt)
        
        loss = (self.loss(input_max_pool, gt_max_pool) + self.loss(input_min_pool, gt_min_pool))/2
        return loss   



def normalize(x):
    gpu_id = x.get_device()
    return (x- mean.cuda(gpu_id))/std.cuda(gpu_id)


class TrainingLoss(object):

    def __init__(self, opt, flowwarpper):
        self.opt = opt
        self.flowwarp = flowwarpper
        self.CELoss = nn.CrossEntropyLoss()

    def celoss(self, seg_pred, seg_gt):
        ce_loss = 0
        for i in range(self.opt.vid_length):
            ce_loss+=self.CELoss(seg_pred[:, i], seg_gt[:, i])
        return ce_loss

    def gdloss(self, a, b):
        xloss = torch.sum(
            torch.abs(torch.abs(a[:, :, 1:, :] - a[:, :, :-1, :]) - torch.abs(b[:, :, 1:, :] - b[:, :, :-1, :])))
        yloss = torch.sum(
            torch.abs(torch.abs(a[:, :, :, 1:] - a[:, :, :, :-1]) - torch.abs(b[:, :, :, 1:] - b[:, :, :, :-1])))
        return (xloss + yloss) / (a.size()[0] * a.size()[1] * a.size()[2] * a.size()[3])

    def vgg_loss(self, y_pred_feat, y_true_feat):
        loss = 0
        for i in range(len(y_pred_feat)):
            loss += (y_true_feat[i] - y_pred_feat[i]).abs().mean()
        return loss/len(y_pred_feat)

    def _quickflowloss(self, flow, img, neighber=5, alpha=1):
        flow_scaled = torch.tensor([128,128]).float().cuda(flow.get_device()).view(1,2,1,1).contiguous()
        flow = flow*flow_scaled
        img = img * 256
        bs, c, h, w = img.size()
        center = int((neighber - 1) / 2)
        loss = []
        neighberrange = list(range(neighber))
        neighberrange.remove(center)
        for i in neighberrange:
            for j in neighberrange:
                flowsub = (flow[:, :, center:-center, center:-center] -
                           flow[:, :, i:h - (neighber - i - 1), j:w - (neighber - j - 1)]) ** 2
                imgsub = (img[:, :, center:-center, center:-center] -
                          img[:, :, i:h - (neighber - i - 1), j:w - (neighber - j - 1)]) ** 2
                flowsub = flowsub.sum(1)
                imgsub = imgsub.sum(1)
                indexsub = (i - center) ** 2 + (j - center) ** 2
                loss.append(flowsub * torch.exp(-alpha * imgsub - indexsub))
        return torch.stack(loss).sum() / (bs * w * h)

    def quickflowloss(self, flow, img, t=1):
        flowloss = 0.
        for ii in range(t):
                flowloss += self._quickflowloss(flow[:, :, ii, :, :], img[:, ii, :, :, :])
        return flowloss

    def _flowgradloss(self, flow, image):
        flow = flow * 128
        image = image * 256
        flowgradx = gradientx(flow)
        flowgrady = gradienty(flow)
        imggradx = gradientx(image)
        imggrady = gradienty(image)
        weightx = torch.exp(-torch.mean(torch.abs(imggradx), 1, keepdim=True))
        weighty = torch.exp(-torch.mean(torch.abs(imggrady), 1, keepdim=True))
        lossx = flowgradx * weightx
        lossy = flowgrady * weighty
        # return torch.mean(torch.abs(lossx + lossy))
        return torch.mean(torch.abs(lossx)) + torch.mean(torch.abs(lossy))

    def flowgradloss(self, flow, image, t=1):

        flow_gradient_loss = 0.
        for ii in range(t):
            flow_gradient_loss += self._flowgradloss(flow[:, :, ii, :, :], image[:, ii, :, :, :])
        return flow_gradient_loss/t

    def imagegradloss(self, input, target):
        input_gradx = ops.gradientx(input)
        input_grady = ops.gradienty(input)

        target_gradx = ops.gradientx(target)
        target_grady = ops.gradienty(target)

        return F.l1_loss(torch.abs(target_gradx), torch.abs(input_gradx)) \
               + F.l1_loss(torch.abs(target_grady), torch.abs(input_grady))

    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = F.avg_pool2d(x, 3, 1)
        mu_y = F.avg_pool2d(y, 3, 1)

        sigma_x = F.avg_pool2d(x ** 2, 3, 1) - mu_x ** 2
        sigma_y = F.avg_pool2d(y ** 2, 3, 1) - mu_y ** 2
        sigma_xy = F.avg_pool2d(x * y, 3, 1) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return torch.clamp((1 - SSIM) / 2, 0, 1).mean()

    def image_similarity(self, x, y, opt=None):
        sim = 0
        vid_len = x.size(1)
        # for ii in range(opt.vid_length):
        for ii in range(vid_len):
            sim += 1 * self.SSIM(x[:, ii, ...], y[:, ii, ...]) \
                  + (1 - 1) * F.l1_loss(x[:, ii, ...], y[:, ii, ...])
        return sim/vid_len
    
    def loss_function(self, mu, logvar, bs):
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return KLD/bs

    def kl_criterion(self, mu, logvar, bs):
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD /= self.opt.batch_size
        return KLD

    def _flowconsist(self, flow, flowback, mask_fw=None, mask_bw=None):
        if mask_fw is not None:
            # mask_fw, mask_bw = occlusion(flow, flowback, self.flowwarp)
            prevloss = (mask_bw * torch.abs(self.flowwarp(flow, -flowback) - flowback)).mean()
            nextloss = (mask_fw * torch.abs(self.flowwarp(flowback, flow) - flow)).mean()
        else:
            prevloss = torch.abs(self.flowwarp(flow, -flowback) - flowback).mean()
            nextloss = torch.abs(self.flowwarp(flowback, flow) - flow).mean()
        return prevloss + nextloss

    def flowconsist(self, flow, flowback, mask_fw=None, mask_bw=None, t=4):
        flowcon = 0.
        if mask_bw is not None:
            for ii in range(t):
                flowcon += self._flowconsist(flow[:, :, ii, :, :], flowback[:, :, ii, :, :],
                                                 mask_fw=mask_fw[:, ii:ii + 1, ...],
                                                 mask_bw=mask_bw[:, ii:ii + 1, ...])
        else:
            for ii in range(t):
                flowcon += self._flowconsist(flow[:, :, ii, :, :], flowback[:, :, ii, :, :])
        return flowcon

    def reconlossT(self, x, y, t=4, mask=None):
        if mask is not None:
            x = x * mask.unsqueeze(2)
            y = y * mask.unsqueeze(2)

        loss = (x.contiguous() - y.contiguous()).abs().mean()
        return loss


class losses_multigpu_only_mask(nn.Module):
    def __init__(self, opt, flowwarpper):
        super(losses_multigpu_only_mask, self).__init__()
        self.tl = TrainingLoss(opt, flowwarpper)
        self.flowwarpper = flowwarpper
        self.opt = opt

    def forward(self, rgb_data, y_pred, mu, logvar, flow, flowback, mask_fw, mask_bw, prediction_vgg_feature, gt_vgg_feature,  y_pred_before_refine=None, \
                    seg_pred=None, seg_data=None):
        '''
            frame1 (bs, 3, 128, 128)
            frame2 (bs, 4, 3, 128, 128)
            y_pred (bs, 4, 3, 128, 128)
            mu     (bs*4, 1024)
            logvar (bs*4, 1024)
            flow   (bs, 2, 4, 128, 128)
            flowbac(bs, 2, 4, 128, 128)
            mask_fw(bs, 4, 128, 128)
            mask_bw(bs, 4, 128, 128)
            y_pred_before_refine (bs, 4, 3, 128, 128)
        '''
        batch_size = y_pred.size(0)
        frame1 = rgb_data[:, 0]
        frame2 = rgb_data[:, 1:]
        opt = self.opt
        flowwarpper = self.flowwarpper
        tl = self.tl
        output = y_pred

        '''flowloss'''
        flowloss = tl.quickflowloss(flow, frame2, t=opt.vid_length)
        flowloss += tl.quickflowloss(flowback, frame1.unsqueeze(1).repeat(1, opt.vid_length, 1,1,1), t=opt.vid_length)
        flowloss *= 0.01

        '''flow consist'''
        flowcon = tl.flowconsist(flow, flowback, mask_fw, mask_bw, t=opt.vid_length)

        '''kldloss'''
        kldloss = tl.loss_function(mu, logvar, batch_size)

        '''flow gradient loss'''
        # flow_gradient_loss = tl.flowgradloss(flow, frame2)
        # flow_gradient_loss += tl.flowgradloss(flowback, frame1)
        # flow_gradient_loss *= 0.01

        '''Image Similarity loss'''
        sim_loss = tl.image_similarity(output, frame2, opt)

        '''reconstruct loss'''
        prevframe = [torch.unsqueeze(flowwarpper(frame2[:, ii, :, :, :], -flowback[:, :, ii, :, :]* mask_bw[:, ii:ii + 1, ...]), 1)
                     for ii in range(opt.vid_length)]
        prevframe = torch.cat(prevframe, 1)

        reconloss_back = tl.reconlossT(prevframe,
                                  torch.unsqueeze(frame1, 1).repeat(1, opt.vid_length, 1, 1, 1),
                                  mask=mask_bw, t=opt.vid_length)
        reconloss = tl.reconlossT(output, frame2, t=opt.vid_length)

        if y_pred_before_refine is not None:
            reconloss_before = tl.reconlossT(y_pred_before_refine, frame2, mask=mask_fw, t=opt.vid_length)
        else:
            reconloss_before = 0.

        '''vgg loss'''
        vgg_loss = tl.vgg_loss(prediction_vgg_feature, gt_vgg_feature)

        '''mask loss'''
        mask_loss = (1 - mask_bw).mean() + (1 - mask_fw).mean()
        d = OrderedDict([
                ('flow', flowloss),
                ('recon',0*reconloss), #2
                ('recon_back', 2*reconloss_back), # 10
                ('recon_before', 2*reconloss_before),
                ('kld', 0.2*kldloss),
                ('flowcon', flowcon),
                ('sim', 0*sim_loss), # 0.5
                ('vgg', 0*vgg_loss), # 0.6
                ('mask', 0.2*mask_loss) # 1
            ])

        if self.opt.seg:
            seg_gt = torch.argmax(seg_data[:,1:], dim=2)
            ce_loss = self.tl.celoss(seg_pred, seg_gt)
            d['ce'] = 0.04*ce_loss

        for i in list(d.keys()):
            d[i]*=10
        return d


        # flowloss, reconloss, reconloss_back, reconloss_before, kldloss, flowcon, sim_loss, vgg_loss, mask_loss


# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python main.py  --disp_interval 100 --mode xs2[247/1964]type inter --bs 48 --nw 8  --s 0 --checksession 0 --checkepoch_range --checkepoch_low 20 --checkepoch_up 30 --val --checkpoint 1735 --load_dir log/MyFRRN_xs2xs_inter_0_05-31-23:01:33  gen