### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import numpy as np
import torch
import torch.nn as nn
import os
from torch.autograd import Variable
from utils.net_utils import *
from .multi_scale_discriminator import MultiscaleDiscriminator
from .motion_discriminator import MotionDiscriminator

from .MyFRRN import MyFRRN 

class GAN(nn.Module):
    def __init__(self,  args):
        super(GAN, self).__init__()
        self.args = args

        if args.netG == 'MyFRRN':
            self.netG = MyFRRN(args)#.cuda()
        if args.netD == 'multi_scale':
            self.netD = MultiscaleDiscriminator(3*3, ndf=64, n_layers=args.n_layer_D, norm_layer=nn.BatchNorm2d, 
                                                                use_sigmoid=True, num_D=args.num_D, getIntermFeat=True)
        if args.netD == 'multi_scale_img':
            self.netD = MultiscaleDiscriminator(3, ndf=64, n_layers=args.n_layer_D, norm_layer=nn.BatchNorm2d, 
                                                                use_sigmoid=True, num_D=args.num_D, getIntermFeat=True)   
        if args.netD == 'multi_scale_img_seg':
            self.netD = MultiscaleDiscriminator(3+1, ndf=64, n_layers=args.n_layer_D, norm_layer=nn.BatchNorm2d, 
                                                                use_sigmoid=True, num_D=args.num_D, getIntermFeat=True)   
        if args.netD == 'motion_img' or args.netD =='motion_img_seg':
            self.netD = MotionDiscriminator(3)              
        self.netG.apply(weights_init)
        self.netD.apply(weights_init)

        # self.optG = torch.optim.Adamax(self.netG.parameters(), lr=opt.g_lr)
        # self.optD = torch.optim.SGD(self.netD.parameters(), lr=opt.d_lr, momentum=0.9)

    def discriminate(self, input, test_image, use_pool=False):
        input_concat = torch.cat((input, test_image.detach()), dim=1)
        if use_pool:            
            fake_query = self.query(input_concat)
            return self.netD(fake_query)
        else:
            return self.netD(input_concat)

    def single_discriminate(self, test_image, use_pool=False):
        input_concat = test_image.detach()
        if use_pool:            
            fake_query = self.query(input_concat)
            return self.netD(fake_query)
        else:
            return self.netD(input_concat)      
    def query(self, fake_vids):
        '''
            use query when you need to redfine variable for discriminator input
            e.g. when fake_vids is output from generator
        '''
        return_images = []
        for image in fake_vids.data:
            image = torch.unsqueeze(image, 0)
            return_images.append(image)
        gpu_id = fake_vids.get_device()
        return_images = Variable(torch.cat(return_images, 0)).cuda(gpu_id)
        return return_images

    def set_net_grad(self, net, flag=True):
        for p in net.parameters():
            p.requires_grad = flag

    def create_disc_label_map(self, real_image, fake_image):
        diff_map = torch.mean(torch.abs(real_image-fake_image), dim=1, keepdim=True)
        # diff_map = diff_map - torch.mean(diff_map)
        # diff_map_std = torch.std(diff_map)
        
        # label_map[diff_map<0] = 1
        # avg to discriminator size
        diff_map = nn.functional.avg_pool2d(diff_map, kernel_size=23, stride=4, padding=11, count_include_pad=False)
        label_map = torch.zeros(diff_map.size()).cuda(fake_image.get_device())
        # print("mean", torch.mean(diff_map))
        # print("std", diff_map_std)
        label_map[diff_map<=0.06] = 1
        # label_map[label_map<=0.7] = 0
        return label_map

    def forward(self, input, gt=None):
        # Fake Generation
        if self.training:
            fake_image, fake_seg, gt_seg_encoded = self.netG(input, gt)
            # print("fake_image", fake_image.requires_grad)
            # print("input", input.requires_grad)
            # print("gt", gt.requires_grad)
            # fake_image = torch.zeros(fake_image.size()).cuda(fake_image.get_device())

            # input = input[:,:6]

            
            if self.args.netD == 'multi_scale':
                # Fake Detection and Loss
                pred_fake_D = self.discriminate(input, fake_image, use_pool=True)      
                # Real Detection and Loss       
                pred_real_D = self.discriminate(input, gt[:, :3])

                # GAN loss (Fake Possibility Loss)     
                self.set_net_grad(self.netD, False)   
                pred_fake_G = self.netD(torch.cat((input, fake_image), dim=1))  
                self.set_net_grad(self.netD, True)
            elif self.args.netD == 'multi_scale_img':
                # Fake Detection and Loss
                pred_fake_D = self.netD(fake_image.detach())  

                # Real Detection and Loss       
                pred_real_D = self.netD(gt[:,:3])
                # GAN loss (Fake Possibility Loss)     
                self.set_net_grad(self.netD, False)   
                pred_fake_G = self.netD(fake_image)
                self.set_net_grad(self.netD, True)

                # create discriminator label map
                label_map = self.create_disc_label_map(gt[:,:3], fake_image)

            elif self.args.netD == 'multi_scale_img_seg':
                fake_input = torch.cat([fake_image, gt_seg_encoded],dim=1)

                # Fake Detection and Loss
                pred_fake_D = self.netD(fake_input.detach())  

                # Real Detection and Loss       
                pred_real_D = self.netD(torch.cat([gt[:,:3], gt_seg_encoded],dim=1))
                # GAN loss (Fake Possibility Loss)     
                self.set_net_grad(self.netD, False)   
                pred_fake_G = self.netD(fake_input)
                self.set_net_grad(self.netD, True)     
                # create discriminator label map
                label_map = self.create_disc_label_map(gt[:,:3], fake_image)
            elif self.args.netD =='motion_img':
                fake_input =   torch.cat([input[:,:3], fake_image, input[:,3:6]], dim=1)
                gt_input = torch.cat([input[:,:3], gt[:,:3], input[:,3:6]], dim=1)

                # Fake Detection and Loss
                pred_fake_D = self.netD(fake_input.detach())  

                # Real Detection and Loss
                pred_real_D = self.netD(gt_input)

                # GAN loss (Fake Possibility Loss)     
                self.set_net_grad(self.netD, False)   
                pred_fake_G = self.netD(fake_input)
                self.set_net_grad(self.netD, True)
                # create discriminator label map
                label_map = self.create_disc_label_map(gt[:,:3], fake_image)
            elif self.args.netD =='motion_img_seg':
                fake_input =   torch.cat([input[:,:3], fake_image, input[:,3:6]], dim=1)
                gt_input = torch.cat([input[:,:3], gt[:,:3], input[:,3:6]], dim=1)
                gt_segs = torch.cat([input[:,6:26], gt[:,3:23], input[:,26:46]], dim=1)
                fake_seg = torch.argmax(fake_seg, dim=1)
                fake_seg = torch.eye(20)[fake_seg].permute(0,3,1,2).contiguous().float().cuda(fake_seg.get_device())*2-1
                # print("fake_seg", fake_seg.size())
                # print('input_size',input.size())
                fake_segs = torch.cat([input[:,6:26], gt[:,3:23], input[:,26:46]], dim=1)
                # print('fake segs', fake_segs.size())
                # Fake Detection and Loss
                pred_fake_D = self.netD(fake_input, fake_segs)  

                # Real Detection and Loss
                pred_real_D = self.netD(gt_input,gt_segs)

                # GAN loss (Fake Possibility Loss)     
                self.set_net_grad(self.netD, False)   
                pred_fake_G = self.netD(fake_input, fake_segs)
                self.set_net_grad(self.netD, True)
                # create discriminator label map
                label_map = self.create_disc_label_map(gt[:,:3], fake_image)

            return fake_image,  fake_seg, pred_fake_D, pred_real_D, pred_fake_G, label_map
        else:
            fake_image, fake_seg, gt_seg_encoded = self.netG(input, gt)
            # input = input[:,:6]
            if self.args.netD == 'multi_scale': 
                pred_fake = self.netD(torch.cat((input, fake_image), dim=1))
                pred_real = self.netD(torch.cat((input, gt[:, :3]), dim=1))
            elif self.args.netD == 'multi_scale_img':
                pred_fake = self.netD(fake_image)
                pred_real = self.netD(gt[:, :3])
            elif self.args.netD == 'motion_img':
                fake_input =   torch.cat([input[:,:3], fake_image, input[:,3:6]], dim=1)
                gt_input = torch.cat([input[:,:3], gt[:,:3], input[:,3:6]], dim=1)
                # gt_segs = torch.cat([input[:,6:26], gt[:,3:23], input[:,26:46]], dim=1)
                # fake_seg = torch.argmax(fake_seg, dim=1)
                # fake_seg = torch.eye(20)[fake_seg].permute(0,3,1,2).contiguous().float().cuda(fake_seg.get_device())*2-1
                # print("fake_seg", fake_seg.size())
                # print('input_size',input.size())
                # fake_segs = torch.cat([input[:,6:26], gt[:,3:23], input[:,26:46]], dim=1)
                # print('fake segs', fake_segs.size())
                # Fake Detection and Loss
                pred_fake = self.netD(fake_input)  

                # Real Detection and Loss
                pred_real = self.netD(gt_input)

                # GAN loss (Fake Possibility Loss)     
                # self.set_net_grad(self.netD, False)   
                # pred_fake_G = self.netD(fake_input, fake_segs)
                # self.set_net_grad(self.netD, True)
                # create discriminator label map
                label_map = self.create_disc_label_map(gt[:,:3], fake_image)            
            return fake_image, fake_seg, pred_fake, pred_real, None, label_map






