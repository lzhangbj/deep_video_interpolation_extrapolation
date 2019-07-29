import os
import sys
from time import time
import math
import argparse
import itertools
import shutil
from collections import OrderedDict
import cv2 
import numpy as np
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.distributed as dist 
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets 
from tensorboardX import SummaryWriter 
from torchvision.utils import make_grid, save_image
# from utils import AverageMeter
# from loss import CombinedLoss
from losses import RGBLoss, PSNR, SSIM, IoU, GANLoss, VGGCosineLoss
import nets

from data import get_dataset
from folder import rgb_load, seg_load
from utils.net_utils import *
import pickle

def get_model(args):
    # build model
    model = nets.__dict__[args.model](args)
    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class InterTrainer:
    def __init__(self, args):
        self.args = args

        args.logger.info('Initializing trainer')
        self.model = get_model(args)
        if not self.args.train_coarse:
            for p in self.model.coarse_model.parameters():
                p.requires_grad = False
        if self.args.refine and not self.args.train_refine:
            for p in self.model.refine_model.parameters():
                p.requires_grad = False

        params_cnt = count_parameters(self.model.coarse_model)
        args.logger.info("coarse params "+str(params_cnt))
        if self.args.refine:
            params_cnt = count_parameters(self.model.refine_model)
            args.logger.info("refine params "+str(params_cnt))
            if self.args.stage3:
                params_cnt = count_parameters(self.model.stage3_model)
                args.logger.info("stage3 params "+str(params_cnt))
        torch.cuda.set_device(args.rank)
        self.model.cuda(args.rank)
        self.model = torch.nn.parallel.DistributedDataParallel(self.model,
                device_ids=[args.rank])
        
        if self.args.split in ['train', 'val']:
            train_dataset, val_dataset = get_dataset(args)

        if args.split == 'train':
            # train loss
            self.RGBLoss = RGBLoss(args)
            if self.args.refine:
                self.refine_RGBLoss = RGBLoss(args, refine=True)
                self.refine_RGBLoss.cuda(args.rank)
            self.SegLoss = nn.CrossEntropyLoss()
            self.RGBLoss.cuda(args.rank)
            self.SegLoss.cuda(args.rank)

            self.coarse_opt = torch.optim.Adamax(list(self.model.module.coarse_model.parameters()), lr=args.coarse_learning_rate)
            if self.args.refine:
                self.refine_opt = torch.optim.Adamax(list(self.model.module.refine_model.parameters()), lr=args.refine_learning_rate)
            if self.args.stage3:
                self.stage3_opt = torch.optim.Adamax(list(self.model.module.stage3_model.parameters()), lr=args.refine_learning_rate)
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            self.train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size//args.gpus, shuffle=False,
                num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)

        elif args.split == 'val':
            # val criteria
            self.L1Loss  = nn.L1Loss().cuda(args.rank)
            self.PSNRLoss = PSNR().cuda(args.rank)
            self.SSIMLoss = SSIM().cuda(args.rank)
            self.IoULoss = IoU().cuda(args.rank)
            self.VGGCosLoss = VGGCosineLoss().cuda(args.rank)

            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
            self.val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=args.batch_size//args.gpus, shuffle=False,
                num_workers=args.num_workers, pin_memory=True, sampler=val_sampler)

        torch.backends.cudnn.benchmark = True
        self.global_step = 0
        self.epoch=1
        if args.resume or ( args.split != 'train' and not args.checkepoch_range) or self.args.load_coarse or self.args.load_refine:
            self.load_checkpoint()

        if args.rank == 0:
            writer_name = args.path+'/{}_int_{}_len_{}_{}_logs'.format(self.args.split, int(self.args.interval), self.args.vid_length, self.args.dataset)
            if self.args.with_gt_seg:
                writer_name = args.path+'/{}_gtseg_int_{}_len_{}_{}_logs'.format(self.args.split, int(self.args.interval), self.args.vid_length, self.args.dataset)
            self.writer = SummaryWriter(writer_name)

        self.stand_heat_map = self.create_stand_heatmap()
        # self.stand_flow_map = self.standard_flow_map()

    def set_epoch(self, epoch):
        self.args.logger.info("Start of epoch %d" % (epoch+1))
        self.epoch = epoch + 1
        self.train_loader.sampler.set_epoch(epoch)

    def get_input(self, data):
        x = torch.cat([data['frame1'], data['frame2']], dim=1)
        seg = torch.cat([data['seg1'], data['seg2']], dim=1) if self.args.mode == 'xs2xs' else None
        gt_x = [ data['frame'+str(i+3)] for i in range(self.args.vid_length) ]
        gt_seg = [ data['seg'+str(i+3)] if self.args.mode == 'xs2xs' else None for i in range(self.args.vid_length) ]
        return x, seg, gt_x, gt_seg

    def normalize(self, img):
        return (img+1)/2

    def create_stand_heatmap(self):
        heatmap = torch.zeros(3, 128, 256)
        for i in range(256):
            heatmap[0, :, i] = max(0, 1 - 2.*i/256)
            heatmap[1, :, i] = max(0, 2.*i/256 - 1)
            heatmap[2, :, i] = 1-heatmap[0, :, i]-heatmap[1, :, i]
        return heatmap

    def create_heatmap(self, prob_map):
        c, h, w = prob_map.size()
        assert c==1, c
        assert h==128, h
        rgb_prob_map = torch.zeros(3, h, w)
        minimum, maximum = 0.0, 1.0
        ratio = 2 * (prob_map-minimum) / (maximum - minimum)

        rgb_prob_map[0] = 1-ratio
        rgb_prob_map[1] = ratio-1
        rgb_prob_map[:2].clamp_(0,1)
        rgb_prob_map[2] = 1-rgb_prob_map[0]-rgb_prob_map[1]
        return rgb_prob_map

    # def standard_flow_map(self):
    #     W = self.args.input_w
    #     H = self.args.input_h
    #     w = self.model.module.stage3_model.w
    #     h = self.model.module.stage3_model.h
    #     if self.args.stage3 and self.args.stage3_model == 'MSResAttnRefineV2':
    #         scale_factors = [16,8,4]
    #     else:
    #         scale_factors = [16,8,4]


    #     offset = torch.zeros(1,2,H,W)
    #     h_unit = float(H)/(2*(h//2)*sum(scale_factors))
    #     w_unit = float(W)/(2*(w//2)*sum(scale_factors))
    #     for i in range(H):
    #         offset[:,1, i, :] = math.floor(i/h_unit)
    #     for i in range(W):
    #         offset[:,0,:, i] = math.floor(i/w_unit)

    #     h_w_add = torch.zeros(1, 2, H, W)
    #     h_w_add[:, 0] = (2*(w//2)*sum(scale_factors))/2
    #     h_w_add[:, 1] = (2*(h//2)*sum(scale_factors))/2
    #     offset = offset - h_w_add

    #     flow_map = self.flow_to_image(flow=offset[0])
    #     return flow_map

    # def flow_to_image(self, flow):
    #     c, h, w = flow.size()
    #     flow_np = flow.permute(1,2,0).contiguous().numpy()
    #     hsv = np.zeros((h, w, 3), dtype=np.uint8)
    #     hsv[..., 1] = 255

    #     mag, ang = cv2.cartToPolar(flow_np[..., 0], flow_np[..., 1])
    #     hsv[..., 0] = ang * 180 / np.pi / 2
    #     hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    #     bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    #     bgr = bgr.astype(np.float32)/255
    #     return torch.tensor(bgr).permute(2,0,1).contiguous()


    # def reconstruct_flow(self, flow_maps):
    #     H1, W1 = list(flow_maps[0].size())[-2:] # 2, 2, H, W
    #     H2, W2 = list(flow_maps[1].size())[-2:]
    #     H3, W3 = list(flow_maps[2].size())[-2:]
    #     if self.args.stage3 and self.args.stage3_model == 'MSResAttnRefineV2':
    #         assert H1 == 8
    #         assert H2 == 16
    #         assert H3 == 32
    #         scale_factors = [16,8,4]
    #     else:
    #         assert H1 == 8
    #         assert H2 == 16
    #         assert H3 == 32
    #         scale_factors = [16,8,4]

    #     for_flow_maps = []
    #     for_flow_maps.append(F.interpolate(flow_maps[0][0].unsqueeze(0), scale_factor=scale_factors[0], mode='bilinear', align_corners=True).squeeze(0)*scale_factors[0])
    #     for_flow_maps.append(F.interpolate(flow_maps[1][0].unsqueeze(0), scale_factor=scale_factors[1], mode='bilinear', align_corners=True).squeeze(0)*scale_factors[1])
    #     for_flow_maps.append(F.interpolate(flow_maps[2][0].unsqueeze(0), scale_factor=scale_factors[2], mode='bilinear', align_corners=True).squeeze(0)*scale_factors[2])
    #     for_flow_maps[1] = for_flow_maps[0] + for_flow_maps[1]
    #     for_flow_maps[2] = for_flow_maps[1] + for_flow_maps[2]

    #     back_flow_maps = []
    #     back_flow_maps.append(F.interpolate(flow_maps[0][1].unsqueeze(0), scale_factor=scale_factors[0], mode='bilinear', align_corners=True).squeeze(0)*scale_factors[0])
    #     back_flow_maps.append(F.interpolate(flow_maps[1][1].unsqueeze(0), scale_factor=scale_factors[1], mode='bilinear', align_corners=True).squeeze(0)*scale_factors[1])
    #     back_flow_maps.append(F.interpolate(flow_maps[2][1].unsqueeze(0), scale_factor=scale_factors[2], mode='bilinear', align_corners=True).squeeze(0)*scale_factors[2])
    #     back_flow_maps[1] = back_flow_maps[0] + back_flow_maps[1]
    #     back_flow_maps[2] = back_flow_maps[1] + back_flow_maps[2]
        
    #     return for_flow_maps + back_flow_maps

    def prepare_image_set(self, data, coarse_img, coarse_seg, refine_imgs=None, refine_seg=None, stage3_imgs=None, flow_maps=None):
        '''
            input unnormalized img and seg cpu 
        '''
        # assert len(imgs) == self.args.num_pred_step*self.args.num_pred_once
        num_pred_imgs = self.args.num_pred_step*self.args.num_pred_once if self.args.syn_type == 'extra' else 3
        view_gt_rgbs = [ self.normalize(data['frame'+str(i+1)][0]) for i in range(3)]
        view_gt_segs = [ vis_seg_mask(data['seg'+str(i+1)][0].unsqueeze(0), 20).squeeze(0) for i in range(3)]

        black_img = torch.zeros_like(view_gt_rgbs[0])

        view_pred_imgs =  [black_img]
        view_pred_imgs += [self.normalize(coarse_img)]

        view_pred_segs =  [black_img]
        view_pred_segs.append(vis_seg_mask(coarse_seg.unsqueeze(0), 20).squeeze(0))

        if self.args.refine:
            if type(refine_imgs) == list:
                for refine_img in refine_imgs:
                    view_pred_imgs.append(self.normalize(refine_img))
                if refine_seg is not None:
                    view_pred_segs.append(vis_seg_mask(refine_seg.unsqueeze(0), 20).squeeze(0))
            else:
                view_pred_imgs.append(self.normalize(refine_imgs)) 
                if refine_seg is not None:
                    view_pred_segs.append(vis_seg_mask(refine_seg.unsqueeze(0), 20).squeeze(0))
            if self.args.stage3:
                if type(stage3_imgs) == list:
                    view_pred_imgs+=[black_img]*2
                    for stage3_img in stage3_imgs:
                        view_pred_imgs.append(self.normalize(stage3_img))
                else:
                    view_pred_imgs.append(self.normalize(stage3_imgs)) 

        n_rows = self.args.n_scales+2
        while len(view_gt_rgbs) < n_rows:
            view_gt_rgbs.append(black_img)
        while len(view_gt_segs) < n_rows:
            view_gt_segs.append(black_img)
        while len(view_pred_imgs) < n_rows:
            view_pred_imgs.append(black_img)
        while len(view_pred_segs) < n_rows:
            view_pred_segs.append(black_img)

        if self.args.stage3 and flow_maps is not None:
            flow_maps = self.reconstruct_flow(flow_maps)
            flow_maps = [self.flow_to_image(flow) for flow in flow_maps]
            flow_maps.insert(0, black_img)
            flow_maps.insert(1, self.stand_flow_map)
            flow_maps.insert(5, black_img)
            flow_maps.insert(6, self.stand_flow_map)
            view_imgs = view_gt_rgbs + view_pred_imgs + flow_maps + view_gt_segs + view_pred_segs
        else:
            view_imgs = view_gt_rgbs + view_pred_imgs + view_gt_segs + view_pred_segs

        view_imgs = [F.interpolate(img.unsqueeze(0), size=(128, 256), mode='bilinear', align_corners=True)[0] for img in view_imgs]
        
        write_in_img = make_grid(view_imgs, nrow=n_rows) 

        return write_in_img

    def get_loss_record_dict(self):
        D = {'data_cnt':0,
            'all_loss_record':0}

        d = {
        'coarse_l1_loss_record'     :0,
        'coarse_ssim_loss_record'   :0,
        'coarse_gdl_loss_record'    :0,
        'coarse_vgg_loss_record'    :0,
        'coarse_all_loss_record'    :0
        }
        if self.args.mode == 'xs2xs':
            d['coarse_ce_loss_record']=0
        D.update(d)
        if self.args.refine:
            if self.args.split == 'train':
                for i in range(self.args.n_scales):
                    d = {
                    'refine_{}_l1_loss_record'.format(str(1/(2**(self.args.n_scales-i-1))))     :0,
                    'refine_{}_ssim_loss_record'.format(str(1/(2**(self.args.n_scales-i-1))))   :0,
                    'refine_{}_gdl_loss_record'.format(str(1/(2**(self.args.n_scales-i-1))))    :0,
                    'refine_{}_vgg_loss_record'.format(str(1/(2**(self.args.n_scales-i-1))))    :0,
                    'refine_{}_all_loss_record'.format(str(1/(2**(self.args.n_scales-i-1))))    :0
                    }
                    D.update(d)
                    if self.args.stage3:
                        d = {
                        'stage3_{}_l1_loss_record'.format(str(1/(2**(self.args.n_scales-i-1))))     :0,
                        'stage3_{}_ssim_loss_record'.format(str(1/(2**(self.args.n_scales-i-1))))   :0,
                        'stage3_{}_gdl_loss_record'.format(str(1/(2**(self.args.n_scales-i-1))))    :0,
                        'stage3_{}_vgg_loss_record'.format(str(1/(2**(self.args.n_scales-i-1))))    :0,
                        'stage3_{}_all_loss_record'.format(str(1/(2**(self.args.n_scales-i-1))))    :0
                        }
                        D.update(d)
            elif self.args.split == 'val':
                d = {
                    'refine_l1_loss_record'     :0,
                    'refine_ssim_loss_record'   :0,
                    'refine_gdl_loss_record'    :0,
                    'refine_vgg_loss_record'    :0,
                    'refine_all_loss_record'    :0
                    }
                D.update(d)
                if self.args.stage3:
                    d = {
                    'stage3_l1_loss_record'     :0,
                    'stage3_ssim_loss_record'   :0,
                    'stage3_gdl_loss_record'    :0,
                    'stage3_vgg_loss_record'    :0,
                    'stage3_all_loss_record'    :0
                    }
                    D.update(d)

        return D

    def update_loss_record_dict(self, record_dict, loss_dict, batch_size):
        record_dict['data_cnt']+=batch_size
        loss_name_list = ['l1', 'gdl', 'ssim', 'vgg']
        if self.args.mode == 'xs2xs':
            loss_name_list.append('ce')
        for loss_name in loss_name_list:
            record_dict['coarse_{}_loss_record'.format(loss_name)] += \
                            batch_size*loss_dict['coarse_{}_loss'.format(loss_name)].item()
            record_dict['coarse_all_loss_record'] += \
                            batch_size*loss_dict['coarse_{}_loss'.format(loss_name)].item()
        if self.args.refine:
            for i in range(self.args.n_scales):
                for loss_name in ['l1', 'gdl', 'ssim', 'vgg']:
                    record_dict['refine_{}_{}_loss_record'.format(str(1/(2**(self.args.n_scales-i-1))), loss_name)] += \
                                    batch_size*loss_dict['refine_{}_{}_loss'.format(str(1/(2**(self.args.n_scales-i-1))), loss_name)].item()
                    record_dict['refine_{}_all_loss_record'.format(str(1/(2**(self.args.n_scales-i-1))))] += \
                                    batch_size*loss_dict['refine_{}_{}_loss'.format(str(1/(2**(self.args.n_scales-i-1))), loss_name)].item()
                    if self.args.stage3:
                        record_dict['stage3_{}_{}_loss_record'.format(str(1/(2**(self.args.n_scales-i-1))), loss_name)] += \
                                        batch_size*loss_dict['stage3_{}_{}_loss'.format(str(1/(2**(self.args.n_scales-i-1))), loss_name)].item()
                        record_dict['stage3_{}_all_loss_record'.format(str(1/(2**(self.args.n_scales-i-1))))] += \
                                        batch_size*loss_dict['stage3_{}_{}_loss'.format(str(1/(2**(self.args.n_scales-i-1))), loss_name)].item()

        record_dict['all_loss_record']+=batch_size*loss_dict['loss_all'].item()
        return record_dict

    def train(self):
        if self.args.rank == 0:
            self.args.logger.info('Training started')
            step_loss_record_dict = self.get_loss_record_dict()
            epoch_loss_record_dict = self.get_loss_record_dict()
        self.model.train()
        end = time()
        load_time = 0
        comp_time = 0
        

        for step, data in enumerate(self.train_loader):
            self.step = step
            load_time += time() - end
            end = time()
            self.global_step += 1

            batch_size = data['frame1'].size(0)
            loss_dict = OrderedDict()
            # 1. get input
            gt_x = data['frame2'].cuda(self.args.rank, non_blocking=True)
            if self.args.mode =='xs2xs':
                gt_seg = data['seg2'].cuda(self.args.rank, non_blocking=True)
            x = torch.cat([data['frame1'], data['frame3']], dim=1).cuda(self.args.rank, non_blocking=True)
            if self.args.mode =='xs2xs':
                seg = torch.cat([data['seg1'], data['seg3']], dim=1).cuda(self.args.rank, non_blocking=True)
            # 2. model training
            if self.args.refine:
                if self.args.mode == 'xs2xs':
                    if not self.args.stage3:
                        coarse_img, coarse_seg, refine_imgs = self.model(x, seg=seg)
                    else:
                        coarse_img, coarse_seg, refine_imgs, stage3_imgs, flow_maps = self.model(x, seg=seg)
                else:
                    coarse_img, refine_imgs = self.model(x, seg=seg)
            else:
                if self.args.mode == 'xs2xs':
                    coarse_img, coarse_seg = self.model(x, seg=seg)
                else:
                    coarse_img = self.model(x, seg=seg)

            # 3. update outputs and store them
            prefix = 'coarse'
            loss_dict.update(self.RGBLoss(coarse_img, gt_x, False, prefix=prefix))
            if self.args.mode == 'xs2xs':
                loss_dict[prefix+'_ce_loss'] = self.args.ce_weight*self.SegLoss(coarse_seg, torch.argmax(gt_seg, dim=1))   
            if self.args.refine:
                for i in range(self.args.n_scales):
                    prefix='refine_'+str(1/(2**(self.args.n_scales-i-1)))
                    refine_gt_x = F.interpolate(gt_x, scale_factor=1/(2**(self.args.n_scales-i-1)), 
                                                    mode='bilinear', align_corners=True) if i!=self.args.n_scales-1 else gt_x
                    loss_dict.update(self.refine_RGBLoss(refine_imgs[i], refine_gt_x, False, prefix=prefix))
                    if self.args.stage3:
                        prefix='stage3_'+str(1/(2**(self.args.n_scales-i-1)))
                        refine_gt_x = F.interpolate(gt_x, scale_factor=1/(2**(self.args.n_scales-i-1)), 
                                                        mode='bilinear', align_corners=True) if i!=self.args.n_scales-1 else gt_x
                        loss_dict.update(self.refine_RGBLoss(stage3_imgs[i], refine_gt_x, False, prefix=prefix))

            loss = 0
            for i in loss_dict.values():
                loss += torch.mean(i)
            loss_dict['loss_all'] = loss
            self.sync(loss_dict)
            # backward pass
            self.coarse_opt.zero_grad() if self.args.train_coarse  else None
            self.refine_opt.zero_grad() if self.args.refine else None
            self.stage3_opt.zero_grad() if self.args.stage3 else None
            loss_dict['loss_all'].backward()
            self.coarse_opt.step()  if self.args.train_coarse  else None
            self.refine_opt.step()  if self.args.train_refine else None
            self.stage3_opt.step()  if self.args.train_stage3 else None
            comp_time += time() - end
            end = time()

            if self.args.rank == 0:
                step_loss_record_dict = self.update_loss_record_dict(step_loss_record_dict, loss_dict, batch_size)
                # add info to tensorboard
                info = {key:value.item() for key,value in loss_dict.items()}
                self.writer.add_scalars("losses", info, self.global_step)

                if self.step % self.args.disp_interval == 0:
                    for key, value in step_loss_record_dict.items():
                        epoch_loss_record_dict[key]+=value

                    if step_loss_record_dict['data_cnt'] != 0:
                        for key, value in step_loss_record_dict.items():
                            if key!='data_cnt':
                                step_loss_record_dict[key] /= step_loss_record_dict['data_cnt']

                    log_main = 'Epoch [{epoch:d}/{tot_epoch:d}][{cur_batch:d}/{tot_batch:d}] load [{load_time:.3f}s] comp [{comp_time:.3f}s] '.format(epoch=self.epoch, tot_epoch=self.args.epochs,
                            cur_batch=self.step+1, tot_batch=len(self.train_loader),
                            load_time=load_time, comp_time=comp_time)
                    log = '\n\tcoarse l1 [{l1:.3f}] vgg [{vgg:.3f}] ssim [{ssim:.3f}] gdl [{gdl:.3f}]'.format(
                            l1=step_loss_record_dict['coarse_l1_loss_record'],
                            vgg=step_loss_record_dict['coarse_vgg_loss_record'],
                            ssim=step_loss_record_dict['coarse_ssim_loss_record'],
                            gdl=step_loss_record_dict['coarse_gdl_loss_record']
                        )
                    if self.args.mode == 'xs2xs':
                        log+= ' ce [{ce:.3f}]'.format(ce=step_loss_record_dict['coarse_ce_loss_record'])
                    log+= ' all [{all:.3f}]'.format(all=step_loss_record_dict['coarse_all_loss_record'])

                    log_main+=log
                    if self.args.refine:
                        for i in range(self.args.n_scales):
                            scale = 1/(2**(self.args.n_scales-i-1))
                            name_list = ['refine']
                            if self.args.stage3:
                                name_list.append('stage3')
                            for name in name_list:
                                log = '\n\t{name}{scale} l1 [{l1:.3f}] vgg [{vgg:.3f}] ssim [{ssim:.3f}] gdl [{gdl:.3f}] all [{all:.3f}]'.format(
                                        name=name,
                                        scale=str(scale),
                                        l1=step_loss_record_dict['{}_{}_l1_loss_record'.format(name, str(scale))],
                                        vgg=step_loss_record_dict['{}_{}_vgg_loss_record'.format(name, str(scale))],
                                        ssim=step_loss_record_dict['{}_{}_ssim_loss_record'.format(name, str(scale))],
                                        gdl=step_loss_record_dict['{}_{}_gdl_loss_record'.format(name, str(scale))],
                                        all=step_loss_record_dict['{}_{}_all_loss_record'.format(name, str(scale))]
                                    )
                                log_main+=log
                    log_main += '\n\t\t\t\t\t\t\tloss total [{:.3f}]'.format(step_loss_record_dict['all_loss_record'])

                    self.args.logger.info(log_main)
                    comp_time = 0
                    load_time = 0

                    if step_loss_record_dict['data_cnt'] != 0:
                        for key, value in step_loss_record_dict.items():
                            step_loss_record_dict[key] = 0

                if self.step % 30 == 0: 
                    if self.args.refine:
                        if self.args.stage3:
                            image_set = self.prepare_image_set(data, coarse_img[0].clamp_(-1,1).cpu(), coarse_seg[0].cpu(), 
                                                            [ refine_img[0].clamp_(-1,1).cpu() for refine_img in refine_imgs],
                                                            stage3_imgs=[ stage3_img[0].clamp_(-1,1).cpu() for stage3_img in stage3_imgs],
                                                            flow_maps=[ flow_map[0].cpu() for flow_map in flow_maps] if flow_maps is not None else None
                                                        )
                        else:
                            image_set = self.prepare_image_set(data, coarse_img[0].clamp_(-1,1).cpu(), coarse_seg[0].cpu(), 
                                                            [ refine_img[0].clamp_(-1,1).cpu() for refine_img in refine_imgs]
                                                        )
                    else:
                        image_set = self.prepare_image_set(data, coarse_img[0].cpu(), coarse_seg[0].cpu())
                    self.writer.add_image('image_{}'.format(self.global_step), image_set, self.global_step)

        if self.args.rank == 0:
            for key, value in step_loss_record_dict.items():
                epoch_loss_record_dict[key]+=value

            if epoch_loss_record_dict['data_cnt'] != 0:
                for key, value in epoch_loss_record_dict.items():
                    if key!='data_cnt':
                        epoch_loss_record_dict[key] /= epoch_loss_record_dict['data_cnt']

            log_main = 'Epoch [{epoch:d}/{tot_epoch:d}]'.format(epoch=self.epoch, tot_epoch=self.args.epochs)

            log = '\n\tcoarse l1 [{l1:.3f}] vgg [{vgg:.3f}] ssim [{ssim:.3f}] gdl [{gdl:.3f}]'.format(
                    l1=epoch_loss_record_dict['coarse_l1_loss_record'],
                    vgg=epoch_loss_record_dict['coarse_vgg_loss_record'],
                    ssim=epoch_loss_record_dict['coarse_ssim_loss_record'],
                    gdl=epoch_loss_record_dict['coarse_gdl_loss_record']
                )
            if self.args.mode == 'xs2xs':
                log+= ' ce [{ce:.3f}]'.format(ce=epoch_loss_record_dict['coarse_ce_loss_record'])
            log+= ' all [{all:.3f}]'.format(all=epoch_loss_record_dict['coarse_all_loss_record'])

            log_main+=log

            if self.args.refine:
                name_list = ['refine']
                if self.args.stage3:
                    name_list.append('stage3')
                for name in name_list:
                    for i in range(self.args.n_scales):
                        scale = 1/(2**(self.args.n_scales-i-1))
                        log = '\n\t{name}{scale} l1 [{l1:.3f}] vgg [{vgg:.3f}] ssim [{ssim:.3f}] gdl [{gdl:.3f}] all [{all:.3f}]'.format(
                                name=name,
                                scale=str(scale),
                                l1=epoch_loss_record_dict['{}_{}_l1_loss_record'.format(name, str(scale))],
                                vgg=epoch_loss_record_dict['{}_{}_vgg_loss_record'.format(name, str(scale))],
                                ssim=epoch_loss_record_dict['{}_{}_ssim_loss_record'.format(name, str(scale))],
                                gdl=epoch_loss_record_dict['{}_{}_gdl_loss_record'.format(name, str(scale))],
                                all=epoch_loss_record_dict['{}_{}_all_loss_record'.format(name, str(scale))]
                            )
                        log_main+=log
            log_main += '\n\t\t\t\t\t\t\tloss total [{:.3f}]'.format(epoch_loss_record_dict['all_loss_record'])
            self.args.logger.info(
                log_main
            )

    def validate(self):
        self.args.logger.info('Validation epoch {} started'.format(self.epoch))
        self.model.eval()

        val_criteria = {}
        criteria_list = ['coarse_l1', 'coarse_psnr', 'coarse_ssim', 'coarse_vgg']
        if self.args.mode == 'xs2xs':
            criteria_list.append('coarse_iou')
        if self.args.refine:
            criteria_list += ['refine_l1', 'refine_psnr', 'refine_ssim', 'refine_vgg']
        for crit in criteria_list:
            val_criteria[crit] = AverageMeter()

        step_losses = OrderedDict()
        with torch.no_grad():
            end = time()
            load_time = 0
            comp_time = 0
            for i, data in enumerate(self.val_loader):
                load_time += time()-end
                end = time()
                self.step=i

                batch_size = data['frame1'].size(0)

                # 1. get input
                gt_x = data['frame2'].cuda(self.args.rank, non_blocking=True)
                if self.args.mode =='xs2xs':
                    gt_seg = data['seg2'].cuda(self.args.rank, non_blocking=True)
                x = torch.cat([data['frame1'], data['frame3']], dim=1).cuda(self.args.rank, non_blocking=True)
                if self.args.mode =='xs2xs':
                    seg = torch.cat([data['seg1'], data['seg3']], dim=1).cuda(self.args.rank, non_blocking=True)
                # 2. model training
                if self.args.refine:
                    if self.args.mode == 'xs2xs':
                        coarse_img, coarse_seg, refine_imgs= self.model(x, seg=seg, gt_seg=gt_seg)
                        refine_img = refine_imgs[-1]
                        coarse_img.clamp_(-1,1)
                        refine_img.clamp_(-1,1)
                    else:
                        coarse_img, refine_img = self.model(x, seg=seg)
                        refine_img = refine_imgs[-1]
                        coarse_img.clamp_(-1,1)
                        refine_img.clamp_(-1,1)
                else:
                    if self.args.mode == 'xs2xs':
                        coarse_img, coarse_seg = self.model(x, seg=seg)
                        coarse_img.clamp_(-1,1)
                    else:
                        coarse_img = self.model(x, seg=seg)
                        coarse_img.clamp_(-1,1)

                # 3. update outputs and store them
                prefix = 'coarse'
                # rgb criteria
                step_losses[prefix+'_l1']    = self.L1Loss(  self.normalize(coarse_img), 
                                                                    self.normalize(gt_x))
                step_losses[prefix+'_psnr']  = self.PSNRLoss(self.normalize(coarse_img), 
                                                                    self.normalize(gt_x))
                step_losses[prefix+'_ssim']  = 1-self.SSIMLoss(  self.normalize(coarse_img), 
                                                                        self.normalize(gt_x))
                step_losses[prefix+'_iou']   =  self.IoULoss(torch.argmax(coarse_seg, dim=1), 
                                                                    torch.argmax(gt_seg, dim=1))
                step_losses[prefix+'_vgg']   =  self.VGGCosLoss( self.normalize(coarse_img), 
                                                                        self.normalize(gt_x), False)  
                if self.args.refine:
                    step_losses['refine_l1']    = self.L1Loss(  self.normalize(refine_img), 
                                                                        self.normalize(gt_x))
                    step_losses['refine_psnr']  = self.PSNRLoss(self.normalize(refine_img), 
                                                                        self.normalize(gt_x))
                    step_losses['refine_ssim']  = 1-self.SSIMLoss(  self.normalize(refine_img), 
                                                                            self.normalize(gt_x))
                    step_losses['refine_vgg']   =  self.VGGCosLoss( self.normalize(refine_img), 
                                                                            self.normalize(gt_x), False)   

                self.sync(step_losses) # sum

                comp_time += time() - end
                end = time()

                # print
                if self.args.rank == 0:
                    for crit in criteria_list:
                        val_criteria[crit].update(step_losses[crit].cpu().item(), batch_size*self.args.gpus)

                    if self.step % self.args.disp_interval == 0:
                        self.args.logger.info(
                            'Epoch [{epoch:d}][{cur_batch:d}/{tot_batch:d}] '
                            'load [{load_time:.3f}s] comp [{comp_time:.3f}s]'.format(
                                epoch=self.epoch, cur_batch=self.step+1, tot_batch=len(self.val_loader),
                                load_time=load_time, comp_time=comp_time
                            )
                        )
                        comp_time = 0
                        load_time = 0
                    if self.step % 3 == 0:
                        if self.args.refine:
                            image_set = self.prepare_image_set(data, coarse_img[0].cpu(), coarse_seg[0].cpu(), refine_img[0].cpu())
                        else:
                            image_set = self.prepare_image_set(data, coarse_img[0].cpu(), coarse_seg[0].cpu())
                        image_name = 'e{}_img_{}'.format(self.epoch, self.step)
                        self.writer.add_image(image_name, image_set, self.step)

        if self.args.rank == 0:
            log_main = '\n######################### Epoch [{epoch:d}] Evaluation Results #########################'.format(epoch=self.epoch)

            log = '\n\tcoarse l1 [{l1:.3f}] vgg [{vgg:.3f}] psnr [{psnr:.3f}] ssim [{ssim:.3f}] iou [{iou:.3f}]'.format(
                    l1=val_criteria['coarse_l1'].avg,
                    vgg=val_criteria['coarse_vgg'].avg,
                    psnr=val_criteria['coarse_psnr'].avg,
                    ssim=val_criteria['coarse_ssim'].avg,
                    iou=val_criteria['coarse_iou'].avg
                )
            log_main+=log
            if self.args.refine:
                log = '\n\trefine l1 [{l1:.3f}] vgg [{vgg:.3f}] psnr [{psnr:.3f}] ssim [{ssim:.3f}]'.format(
                        l1=val_criteria['refine_l1'].avg,
                        vgg=val_criteria['refine_vgg'].avg,
                        psnr=val_criteria['refine_psnr'].avg,
                        ssim=val_criteria['refine_ssim'].avg
                    )
                log_main+=log

            log_main += '\n#####################################################################################\n'

            self.args.logger.info(log_main)

            tfb_info = {key:value.avg for key,value in val_criteria.items()}
            self.writer.add_scalars('val/score', tfb_info, self.epoch)


    def cycgen(self):
        assert self.args.rank == 0 # only allow single worker
        with open('/data/linz/proj/Dataset/Cityscape/load_files/root_clip.pkl', 'rb') as f:
            clips = pickle.load(f)
            clips_dir = clips['val'][:61] # onlye generate 0-60


        save_dir_split = 'extra_int_{}_len_{}_nearest'.format(int(self.args.interval), self.args.vid_length)

        load_img_dir = os.path.join(self.args.cycgen_load_dir, 'rgb')
        save_img_dir = os.path.join(self.args.path, 'cycgen', 'cityscape', 
                                        '{}x{}'.format(self.args.input_h, self.args.input_w),
                                        '{}'.format(save_dir_split), 
                                        'rgb'
                                    )
        load_clip_img_dirs = [os.path.join(load_img_dir, clip_dir[0]) for clip_dir in clips_dir]

        load_seg_dir = os.path.join(self.args.cycgen_load_dir, 'seg')
        save_seg_dir = os.path.join(self.args.path, 'cycgen', 'cityscape', 
                                        '{}x{}'.format(self.args.input_h, self.args.input_w),
                                        '{}'.format(save_dir_split), 
                                        'seg'
                                    )
        save_vis_seg_dir = os.path.join(self.args.path, 'cycgen', 'cityscape', 
                                        '{}x{}'.format(self.args.input_h, self.args.input_w),
                                        '{}'.format(save_dir_split), 
                                        'vis_seg'
                                    )
        load_clip_seg_dirs = [os.path.join(load_seg_dir, clip_dir[0]) for clip_dir in clips_dir]

        first_index     = 0
        second_index    = first_index + int(self.args.interval)

        end = time()
        for clip_ind in range(len(clips_dir)):
            clip_dir = clips_dir[clip_ind][0]
            # load imgs
            load_clip_img_dir = load_clip_img_dirs[clip_ind]
            load_img_files = [  os.path.join(load_clip_img_dir, '{:0>2d}.0.png'.format(first_index)),
                                os.path.join(load_clip_img_dir, '{:0>2d}.0.png'.format(second_index)) ]
            load_imgs = rgb_load(load_img_files)

            # load segs
            load_clip_seg_dir = load_clip_seg_dirs[clip_ind]
            load_seg_files = [  os.path.join(load_clip_seg_dir, '{:0>2d}.0.png'.format(first_index)),
                                os.path.join(load_clip_seg_dir, '{:0>2d}.0.png'.format(second_index)) ]
            load_segs = seg_load(load_seg_files)
            load_segs = [ np.eye(20)[np.array(i)] for i in load_segs ] 

            for i in range(2):
                load_imgs[i] = transforms.functional.to_tensor(load_imgs[i]).unsqueeze(0)
                load_segs[i] = torch.from_numpy(np.transpose(load_segs[i], (2,0,1))).float().unsqueeze(0)

            pred_img_list, pred_seg_list =  self.mini_test(load_imgs, load_segs)
            save_img_list = load_imgs + pred_img_list
            save_seg_list = [seg.argmax(dim=1) for seg in load_segs] + pred_seg_list
            save_vis_seg_list = [vis_seg_mask(torch.eye(20)[seg].permute(0,3,1,2).contiguous(), 20) for seg in save_seg_list]

            save_img_list = [img.squeeze(0) for img in save_img_list]
            save_seg_list = [seg.squeeze(0) for seg in save_seg_list]
            save_vis_seg_list = [seg.squeeze(0) for seg in save_vis_seg_list]

            ### set pred image save dirs ###
            # rgb dir
            input_imgs = load_imgs
            save_img_prefix = os.path.join(save_img_dir, clip_dir)
            if not os.path.exists(save_img_prefix):
                os.makedirs(save_img_prefix)
            # seg dir
            input_segs = load_segs
            save_seg_prefix = os.path.join(save_seg_dir, clip_dir)
            if not os.path.exists(save_seg_prefix):
                os.makedirs(save_seg_prefix)
            # vis_seg_dir
            input_segs = load_segs
            save_vis_seg_prefix = os.path.join(save_vis_seg_dir, clip_dir)
            if not os.path.exists(save_vis_seg_prefix):
                os.makedirs(save_vis_seg_prefix)

            save_indexes = ['{:0>2d}.0'.format(int(first_index+i*self.args.interval)) for i in range(self.args.vid_length+2)]             

            save_imgs_name = [ os.path.join(save_img_prefix, pred_index+".png") for pred_index in save_indexes ]
            save_segs_name = [ os.path.join(save_seg_prefix, pred_index+".png") for pred_index in save_indexes ]
            save_vis_segs_name = [ os.path.join(save_vis_seg_prefix, pred_index+".png") for pred_index in save_indexes ]

            for i in range(self.args.vid_length + 2):
                save_image(save_img_list[i], save_imgs_name[i])
                save_image(save_seg_list[i], save_segs_name[i])
                save_image(save_vis_seg_list[i], save_vis_segs_name[i])

            p_time = time() - end
            end = time()
            sys.stdout.write('\rprocessing {}/{} {}s {}'.format(clip_ind+1, 61, p_time, clip_dir))


    def mini_test(self, img_list, seg_list):
        '''
            input: cpu tensors
                img_list: list of image tensor of size (bs, 3, h, w) in range [0, 1] 
                seg_list: list of seg tensor of size (bs, 20, h, w)  in range [0, 1]     binary value
            return: cpu tensors
                img_list: list of pred image tensor of size (bs, 3, h, w) in range [0, 1]
                seg_list: list of seg tensor of size (bs, h, w) in range [0, 19]
        '''
        assert self.args.rank == 0, 'only allow single gpu test'
        assert len(img_list) == 2
        assert len(seg_list) == 2

        if len(list(seg_list[0].size())) == 3:
            seg_list = [torch.eye(20)[seg].permute(0,3,1,2).contiguous() for seg in seg_list] 

        # self.args.logger.info('testing started')
        self.model.eval()

        pred_img_list = []
        pred_seg_list = []

        with torch.no_grad():
            end = time()
            load_time = 0
            comp_time = 0

            input_img1 = img_list[0].cuda(self.args.rank, non_blocking=True)*2 - 1
            input_img2 = img_list[1].cuda(self.args.rank, non_blocking=True)*2 - 1
            input_seg1 = seg_list[0].cuda(self.args.rank, non_blocking=True)
            input_seg2 = seg_list[1].cuda(self.args.rank, non_blocking=True)
            for i in range(self.args.num_pred_step):
                input_imgs = torch.cat([input_img1, input_img2], dim=1)
                input_segs = torch.cat([input_seg1, input_seg2], dim=1)
                load_time += time()-end
                end = time()
                self.step=i
                # forward pass
                if self.args.refine:
                    coarse_img, seg, refine_mask, img = self.model(input_imgs, input_segs)

                else:
                    img, seg = self.model(input_imgs, input_segs)       

                for j in range(self.args.num_pred_once):
                    pred_img_list.append(self.normalize(img[:, 3*j:3*j+3]))
                    pred_seg_list.append(torch.argmax(seg[:, 20*j:20*j+20], dim=1))

                # update input image and seg for next step
                if self.args.num_pred_once == 1:
                    input_img1 = input_img2
                    input_img2 = pred_img_list[-1]*2 - 1 
                    input_seg1 = input_seg2
                    input_seg2 = torch.eye(20)[pred_seg_list[-1]].permute(0,3,1,2).contiguous().cuda(self.args.rank, non_blocking=True)
                else:
                    input_img1 = pred_img_list[-2]*2 - 1 
                    input_img2 = pred_img_list[-1]*2 - 1 
                    input_seg1 = torch.eye(20)[pred_seg_list[-2]].permute(0,3,1,2).contiguous().cuda(self.args.rank, non_blocking=True)
                    input_seg2 = torch.eye(20)[pred_seg_list[-1]].permute(0,3,1,2).contiguous().cuda(self.args.rank, non_blocking=True)

                comp_time += time() - end
                end = time()

                comp_time = 0
                load_time = 0

            # self.args.logger.info('testing done')
            pred_img_list = [img.cpu() for img in pred_img_list]
            pred_seg_list = [seg.cpu() for seg in pred_seg_list]

            return pred_img_list, pred_seg_list


    def sync(self, loss_dict, mean=True):
        '''Synchronize all tensors given using mean or sum.'''
        for tensor in loss_dict.values():
            dist.all_reduce(tensor)
            if mean:
                tensor.div_(self.args.gpus)


    def save_checkpoint(self):
        save_md_dir = '{}_{}_{}_{}'.format(self.args.model, self.args.mode, self.args.syn_type, self.args.session)
        save_name = os.path.join(self.args.path, 
                                'checkpoint',
                                save_md_dir + '_{}_{}.pth'.format(self.epoch, self.step))
        self.args.logger.info('Saving checkpoint..')
        save_dict = {
            'session': self.args.session,
            'epoch': self.epoch + 1,
            'coarse_model': self.model.module.coarse_model.state_dict(),
            'coarse_opt': self.coarse_opt.state_dict(),
        }
        if self.args.refine:
            save_dict['refine_model'] = self.model.module.refine_model.state_dict()
            save_dict['refine_opt'] = self.refine_opt.state_dict()
            if self.args.stage3:
                save_dict['stage3_model'] = self.model.module.stage3_model.state_dict()
                save_dict['stage3_opt'] = self.stage3_opt.state_dict()
        torch.save(save_dict, save_name)
        self.args.logger.info('save model: {}'.format(save_name))


    def load_checkpoint(self):
        load_md_dir = '{}_{}_{}_{}'.format(self.args.load_model, self.args.mode, self.args.syn_type, self.args.checksession)
        if self.args.load_dir is not None:
            load_name = os.path.join(self.args.load_dir,
                                    'checkpoint',
                                    load_md_dir+'_{}_{}.pth'.format(self.args.checkepoch, self.args.checkpoint))
        else:
            load_name = os.path.join(load_md_dir+'_{}_{}.pth'.format(self.args.checkepoch, self.args.checkpoint))
        self.args.logger.info('Loading checkpoint %s' % load_name)
        device = torch.device('cpu')
        ckpt = torch.load(load_name, map_location=device)
        
        # load model parameters first
        if self.args.load_coarse:
            new_ckpt = OrderedDict()
            coarse_model_dict = self.model.module.coarse_model.state_dict()
            for key,item in ckpt['coarse_model'].items():
                new_ckpt[key] = item
            coarse_model_dict.update(new_ckpt)
            self.model.module.coarse_model.load_state_dict(coarse_model_dict)
        if self.args.load_refine:
            new_ckpt = OrderedDict()
            assert self.args.refine
            refine_model_dict = self.model.module.refine_model.state_dict()
            for key,item in ckpt['refine_model'].items():
                new_ckpt[key] = item
            refine_model_dict.update(new_ckpt)
            self.model.module.refine_model.load_state_dict(refine_model_dict)
        
        if self.args.load_stage3:
            new_ckpt = OrderedDict()
            assert self.args.stage3
            stage3_model_dict = self.model.module.stage3_model.state_dict()
            for key,item in ckpt['stage3_model'].items():
                new_ckpt[key] = item
            stage3_model_dict.update(new_ckpt)
            self.model.module.stage3_model.load_state_dict(stage3_model_dict)
        

        # load opt
        if self.args.split == 'train':
            # load coarse opt
            if self.args.train_coarse and self.args.load_coarse:
                self.coarse_opt.load_state_dict(ckpt['coarse_opt'])
                for state in self.coarse_opt.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda(self.args.rank)
            # load refine opt
            if self.args.train_refine and self.args.load_refine:
                assert self.args.refine
                self.refine_opt.load_state_dict(ckpt['refine_opt'])
                for state in self.refine_opt.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda(self.args.rank)
            # load stage3 opt
            if self.args.train_stage3 and self.args.load_stage3:
                assert self.args.stage3
                self.stage3_opt.load_state_dict(ckpt['stage3_opt'])
                for state in self.stage3_opt.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda(self.args.rank)

        if self.args.resume:
            assert ckpt['epoch']-1 == self.args.checkepoch, [ckpt['epoch'], self.args.checkepoch]
            self.epoch = ckpt['epoch']
        elif self.args.split != 'train':
            assert ckpt['epoch']-1 == self.args.checkepoch, [ckpt['epoch'], self.args.checkepoch]
            self.epoch = ckpt['epoch'] - 1
        self.args.logger.info('checkpoint loaded')

