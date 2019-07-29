import os
import sys
from time import time
import math
import argparse
import itertools
import shutil
from collections import OrderedDict

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


class ExtraTrainer:
    def __init__(self, args):
        self.args = args

        args.logger.info('Initializing trainer')
        self.model = get_model(args)
        if not self.args.train_coarse:
            for p in self.model.coarse_model.parameters():
                p.requires_grad = False
        if self.args.inpaint and not self.args.train_inpaint:
            for p in self.model.inpaint_model.parameters():
                p.requires_grad = False

        params_cnt = count_parameters(self.model)
        args.logger.info("params "+str(params_cnt))
        torch.cuda.set_device(args.rank)
        self.model.cuda(args.rank)
        self.model = torch.nn.parallel.DistributedDataParallel(self.model,
                device_ids=[args.rank])
        
        if self.args.split in ['train', 'val']:
            train_dataset, val_dataset = get_dataset(args)

        if args.split == 'train':
            # train loss
            self.RGBLoss = RGBLoss(args, sharp=False)
            self.SegLoss = nn.CrossEntropyLoss()
            self.RGBLoss.cuda(args.rank)
            self.SegLoss.cuda(args.rank)

            self.coarse_opt = torch.optim.Adamax(list(self.model.module.coarse_model.parameters()), lr=args.coarse_learning_rate)
            if self.args.inpaint:
                self.inpaint_opt = torch.optim.Adamax(list(self.model.module.inpaint_model.parameters()), lr=args.inpaint_learning_rate)

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
        if args.resume or ( args.split != 'train' and not args.checkepoch_range) or self.args.load_coarse or self.args.load_inpaint:
            self.load_checkpoint()

        if args.rank == 0:
            writer_name = args.path+'/{}_int_{}_len_{}_{}_logs'.format(self.args.split, int(self.args.interval), self.args.vid_length, self.args.dataset)
            self.writer = SummaryWriter(writer_name)

        self.stand_heat_map = self.create_stand_heatmap()

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

    def prepare_image_set(self, data, imgs, segs, masks=None, inpaints=None):
        '''
            input unnormalized img and seg cpu 
        '''
        assert len(imgs) == self.args.num_pred_step*self.args.num_pred_once
        num_pred_imgs = self.args.num_pred_step*self.args.num_pred_once
        view_gt_rgbs = [ self.normalize(data['frame'+str(i+1)][0]) for i in range(num_pred_imgs+2)]
        view_gt_segs = [ vis_seg_mask(data['seg'+str(i+1)][0].unsqueeze(0), 20).squeeze(0) for i in range(num_pred_imgs+2)]

        black_img = torch.zeros_like(view_gt_rgbs[0])

        view_pred_imgs =  [black_img]*2
        view_pred_imgs += [self.normalize(img) for img in imgs]

        view_pred_segs =  [black_img]*2
        view_pred_segs += [vis_seg_mask(seg.unsqueeze(0), 20).squeeze(0) for seg in segs]

        view_imgs = view_gt_rgbs + view_pred_imgs + view_gt_segs + view_pred_segs

        if self.args.inpaint:
            view_inpaint_imgs =  [black_img]*2
            view_inpaint_imgs += [self.normalize(img) for img in inpaints]

            view_inpaint_masks =  [black_img, self.stand_heat_map]
            view_inpaint_masks += [ self.create_heatmap(img) for img in masks]

            view_imgs+=view_inpaint_imgs
            view_imgs+=view_inpaint_masks

        view_imgs = [F.interpolate(img.unsqueeze(0), size=(128, 256), mode='bilinear', align_corners=True)[0] for img in view_imgs]
        write_in_img = make_grid(view_imgs, nrow=num_pred_imgs+2)

        return write_in_img


    def get_loss_record_dict(self,prefix=''):
        D = {'{}_data_cnt'.format(prefix):0,
            '{}_all_loss_record'.format(prefix):0}
        for i in range(self.args.num_pred_step*self.args.num_pred_once):
            d = {
            '{}_frame_{}_coarse_l1_loss_record'.format(prefix, i+1):0,
            '{}_frame_{}_coarse_ssim_loss_record'.format(prefix, i+1):0,
            '{}_frame_{}_coarse_gdl_loss_record'.format(prefix, i+1):0,
            '{}_frame_{}_coarse_vgg_loss_record'.format(prefix, i+1):0,
            '{}_frame_{}_coarse_all_loss_record'.format(prefix, i+1):0
            }
            if self.args.mode == 'xs2xs':
                d['{}_frame_{}_coarse_ce_loss_record'.format(prefix, i+1)]:0
            D.update(d)
            if self.args.inpaint:
                d = {
                '{}_frame_{}_inpaint_l1_loss_record'.format(prefix, i+1):0,
                '{}_frame_{}_inpaint_gdl_loss_record'.format(prefix, i+1):0,
                '{}_frame_{}_inpaint_mask_loss_record'.format(prefix, i+1):0,
                '{}_frame_{}_inpaint_all_loss_record'.format(prefix, i+1):0
                }
                D.update(d)
        return D

    def update_loss_record_dict(self, record_dict, loss_dict, batch_size):
        record_dict['step_data_cnt']+=batch_size
        for i in range(self.args.num_pred_step):
            for j in range(self.args.num_pred_once):
                frame_ind = 1+i*self.args.num_pred_once + j
                loss_name_list = ['l1', 'gdl', 'ssim', 'vgg']
                if self.args.mode == 'xs2xs':
                    loss_name_list.append('ce')
                for loss_name in loss_name_list:
                    record_dict['step_frame_{}_coarse_{}_loss_record'.format(frame_ind, loss_name)] += \
                                    batch_size*loss_dict['step_{}_frame_{}_coarse_{}_loss'.format(i+1, j+1, loss_name)].item()
                    record_dict['step_frame_{}_coarse_all_loss_record'.format(frame_ind)] += \
                                    batch_size*loss_dict['step_{}_frame_{}_coarse_{}_loss'.format(i+1, j+1, loss_name)].item()
                if self.args.inpaint:
                    frame_ind = 1+i*self.args.num_pred_once + j
                    for loss_name in ['l1', 'gdl', 'mask']:
                        record_dict['step_frame_{}_inpaint_{}_loss_record'.format(frame_ind, loss_name)] += \
                                        batch_size*loss_dict['step_{}_frame_{}_inpaint_{}_loss'.format(i+1, j+1, loss_name)].item()
                        record_dict['step_frame_{}_inpaint_all_loss_record'.format(frame_ind)] += \
                                        batch_size*loss_dict['step_{}_frame_{}_inpaint_{}_loss'.format(i+1, j+1, loss_name)].item()
        record_dict['step_all_loss_record']+=batch_size*loss_dict['loss_all'].item()
        return record_dict

    def train(self):
        if self.args.rank == 0:
            self.args.logger.info('Training started')
            step_loss_record_dict = self.get_loss_record_dict('step')
            epoch_loss_record_dict = self.get_loss_record_dict('epoch')
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
            if self.step % 30 == 0 and self.args.rank == 0:  # visualize
                vis_coarse_imgs = []
                vis_inpaint_imgs = []
                vis_inpaint_masks = []
                if self.args.mode == 'xs2xs':
                    vis_coarse_segs = []
            last_rgb_output = torch.cat([data['frame1'], data['frame2']], dim=1).cuda(self.args.rank, non_blocking=True)
            if self.args.mode == 'xs2xs':
                last_seg_output = torch.cat([data['seg1'], data['seg2']], dim=1).cuda(self.args.rank, non_blocking=True)
            if self.args.num_pred_step > 1:
                assert self.args.num_pred_once == 1
            for ii in range(self.args.num_pred_step):
                # 1. update input
                gt_start_frame_ind = 3+ii*self.args.num_pred_once
                gt_x = torch.cat([data['frame'+str(i)] 
                                     for i in range(gt_start_frame_ind, gt_start_frame_ind+self.args.num_pred_once)], dim=1)\
                                            .cuda(self.args.rank, non_blocking=True)
                if self.args.mode =='xs2xs':
                    gt_seg = gt_rgb = torch.cat([data['seg'+str(i)] 
                                         for i in range(gt_start_frame_ind, gt_start_frame_ind+self.args.num_pred_once)], dim=1)\
                                                .cuda(self.args.rank, non_blocking=True)
                x = last_rgb_output
                if self.args.mode =='xs2xs':
                    seg = last_seg_output
                if self.args.fix_init_frames:
                    x = torch.cat([data['frame2'].detach().cuda(self.args.rank, non_blocking=True), x], dim=1)
                    if self.args.mode == 'xs2xs':
                        seg = torch.cat([data['seg2'].detach().cuda(self.args.rank, non_blocking=True), seg], dim=1)

                # 2. model training
                if self.args.inpaint:
                    if self.args.mode == 'xs2xs':
                        coarse_img, coarse_seg, inpaint_mask, inpaint_img = self.model(x, seg=seg, gt_x=gt_x, gt_seg=gt_seg)
                    else:
                        coarse_img, inpaint_mask, inpaint_img = self.model(x, seg=seg, gt_x=gt_x)
                else:
                    if self.args.mode == 'xs2xs':
                        coarse_img, coarse_seg = self.model(x, seg=seg, gt_x=gt_x, gt_seg=gt_seg)
                    else:
                        coarse_img = self.model(x, seg=seg, gt_x=gt_x)

                # 3. update outputs and store them
                for j in range(self.args.num_pred_once):
                    prefix='step_{}_frame_{}_coarse'.format(ii+1,j+1)
                    loss_dict.update(self.RGBLoss(coarse_img[:,j*3:j*3+3], gt_x[:,j*3:j*3+3], False, prefix=prefix))
                    if self.args.mode == 'xs2xs':
                        loss_dict[prefix+'_ce_loss'] = self.args.ce_weight*self.SegLoss(coarse_seg[:,j*20:j*20+20], torch.argmax(gt_seg[:,j*20:j*20+20], dim=1))   
                    if self.step % 30 == 0 and self.args.rank == 0: 
                        vis_coarse_imgs.append(coarse_img[0,j*3:j*3+3].cpu())
                        if self.args.mode =='xs2xs':
                            vis_coarse_segs.append(coarse_seg[0,j*20:j*20+20].cpu())

                    if self.args.inpaint:
                        prefix='step_{}_frame_{}_inpaint'.format(ii+1,j+1)
                        loss_dict.update(self.RGBLoss(inpaint_img[:,j*3:j*3+3]*(1-inpaint_mask[:,j:j+1]), gt_x[:,j*3:j*3+3]*(1-inpaint_mask[:,j:j+1]), False, prefix=prefix, mask=inpaint_mask[:,j:j+1]))
                        mask_loss_co = 80 if self.args.inpaint_mask else 0
                        loss_dict[prefix+'_mask_loss'] = mask_loss_co*inpaint_mask[:,j:j+1].mean()
                        if self.step % 30 == 0 and self.args.rank == 0: 
                            vis_inpaint_masks.append(inpaint_mask[0,j:j+1].cpu())
                            vis_inpaint_imgs.append(inpaint_img[0,j*3:j*3+3].cpu())

                if self.args.num_pred_step == 1:
                    break
                back_img = inpainted_img if self.args.inpaint else out_img
                last_rgb_output = torch.cat( [ x[:,-3:], back_img ], dim=1) 
                if self.args.mode == 'xs2xs':
                    last_seg_output = torch.cat( [ seg[:,-20:], 
                                                torch.eye(20)[out_seg.argmax(dim=1)].permute(0,3,1,2).contiguous().cuda(self.args.rank, non_blocking=True)], dim=1)

            loss = 0
            for i in loss_dict.values():
                loss += torch.mean(i)
            loss_dict['loss_all'] = loss
            
            self.sync(loss_dict)
            # backward pass
            self.coarse_opt.zero_grad() 
            self.inpaint_opt.zero_grad() if self.args.inpaint else None
            loss_dict['loss_all'].backward()
            self.coarse_opt.step()  if self.args.train_coarse  else None
            self.inpaint_opt.step() if self.args.train_inpaint else None
            comp_time += time() - end
            end = time()

            if self.args.rank == 0:
                step_loss_record_dict = self.update_loss_record_dict(step_loss_record_dict, loss_dict, batch_size)
                # add info to tensorboard
                info = {key:value.item() for key,value in loss_dict.items()}
                self.writer.add_scalars("losses", info, self.global_step)

                if self.step % self.args.disp_interval == 0:
                    for key, value in step_loss_record_dict.items():
                        epoch_key = key.replace('step', 'epoch')
                        epoch_loss_record_dict[epoch_key]+=value

                    if step_loss_record_dict['step_data_cnt'] != 0:
                        for key, value in step_loss_record_dict.items():
                            if key!='step_data_cnt':
                                step_loss_record_dict[key] /= step_loss_record_dict['step_data_cnt']

                    log_main = 'Epoch [{epoch:d}/{tot_epoch:d}][{cur_batch:d}/{tot_batch:d}] load [{load_time:.3f}s] comp [{comp_time:.3f}s] '.format(epoch=self.epoch, tot_epoch=self.args.epochs,
                            cur_batch=self.step+1, tot_batch=len(self.train_loader),
                            load_time=load_time, comp_time=comp_time)
                    for i in range(self.args.num_pred_once*self.args.num_pred_step):
                        log = '\n\tframe{ind:.0f} l1 [{l1:.3f}] vgg [{vgg:.3f}] ssim [{ssim:.3f}] gdl [{gdl:.3f}] rgb_all [{rgb_all:.3f}] '.format(
                                ind=i+1, 
                                l1=step_loss_record_dict['step_frame_{}_coarse_l1_loss_record'.format(i+1)],
                                vgg=step_loss_record_dict['step_frame_{}_coarse_vgg_loss_record'.format(i+1)],
                                ssim=step_loss_record_dict['step_frame_{}_coarse_ssim_loss_record'.format(i+1)],
                                gdl=step_loss_record_dict['step_frame_{}_coarse_gdl_loss_record'.format(i+1)],
                                rgb_all=step_loss_record_dict['step_frame_{}_coarse_all_loss_record'.format(i+1)]
                            )
                        if self.args.mode == 'xs2xs':
                            log+= 'ce [{ce:.3f}]'.format(ce=step_loss_record_dict['step_frame_{}_coarse_ce_loss_record'.format(i+1)])
                        log_main+=log
                        if self.args.inpaint:
                            log = ' inp l1 [{l1:.3f}] gdl [{gdl:.3f}] mask [{mask:.3f}] all [{inp_all:.3f}]'.format(
                                    l1=step_loss_record_dict['step_frame_{}_inpaint_l1_loss_record'.format(i+1)],
                                    gdl=step_loss_record_dict['step_frame_{}_inpaint_gdl_loss_record'.format(i+1)],
                                    mask=step_loss_record_dict['step_frame_{}_inpaint_mask_loss_record'.format(i+1)],
                                    inp_all=step_loss_record_dict['step_frame_{}_inpaint_all_loss_record'.format(i+1)]
                                )
                            log_main+=log
                    log_main += '\n\t\t\t\t\tloss total [{:.3f}]'.format(step_loss_record_dict['step_all_loss_record'])

                    self.args.logger.info(
                        log_main
                    )
                    comp_time = 0
                    load_time = 0

                    if step_loss_record_dict['step_data_cnt'] != 0:
                        for key, value in step_loss_record_dict.items():
                            step_loss_record_dict[key] = 0

                if self.step % 30 == 0: 
                    image_set = self.prepare_image_set(data, vis_coarse_imgs, vis_coarse_segs, vis_inpaint_masks, vis_inpaint_imgs)
                    self.writer.add_image('image_{}'.format(self.global_step), image_set, self.global_step)

        if self.args.rank == 0:
            for key, value in step_loss_record_dict.items():
                epoch_key = key.replace('step', 'epoch')
                epoch_loss_record_dict[epoch_key]+=value

            if epoch_loss_record_dict['epoch_data_cnt'] != 0:
                for key, value in epoch_loss_record_dict.items():
                    if key!='epoch_data_cnt':
                        epoch_loss_record_dict[key] /= epoch_loss_record_dict['epoch_data_cnt']

            log_main = 'Epoch [{epoch:d}/{tot_epoch:d}]'.format(epoch=self.epoch, tot_epoch=self.args.epochs)

            for i in range(self.args.num_pred_once*self.args.num_pred_step):
                log = '\n\tframe {ind:.0f} l1 [{l1:.3f}] vgg [{vgg:.3f}] ssim [{ssim:.3f}] gdl [{gdl:.3f}] rgb_all [{rgb_all:.3f}]'.format(
                        ind=1+i, 
                        l1=epoch_loss_record_dict['epoch_frame_{}_coarse_l1_loss_record'.format(i+1)],
                        vgg=epoch_loss_record_dict['epoch_frame_{}_coarse_vgg_loss_record'.format(i+1)],
                        ssim=epoch_loss_record_dict['epoch_frame_{}_coarse_ssim_loss_record'.format(i+1)],
                        gdl=epoch_loss_record_dict['epoch_frame_{}_coarse_gdl_loss_record'.format(i+1)],
                        rgb_all=epoch_loss_record_dict['epoch_frame_{}_coarse_all_loss_record'.format(i+1)]
                    )
                if self.args.mode == 'xs2xs':
                    log+=' ce [{ce:.3f}]'.format(ce=epoch_loss_record_dict['epoch_frame_{}_coarse_ce_loss_record'.format(i+1)])
                log_main+=log
                if self.args.inpaint:
                    log = ' inp l1 [{l1:.3f}] gdl [{gdl:.3f}] mask [{mask:.3f}] all [{inp_all:.3f}]'.format(
                            l1=epoch_loss_record_dict['epoch_frame_{}_inpaint_l1_loss_record'.format(i+1)],
                            gdl=epoch_loss_record_dict['epoch_frame_{}_inpaint_gdl_loss_record'.format(i+1)],
                            mask=epoch_loss_record_dict['epoch_frame_{}_inpaint_mask_loss_record'.format(i+1)],
                            inp_all=epoch_loss_record_dict['epoch_frame_{}_inpaint_all_loss_record'.format(i+1)]
                        )
                    log_main+=log
            log_main += '\n\t\t\t\t\t\t\tloss total [{:.3f}]'.format(epoch_loss_record_dict['epoch_all_loss_record'])

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
        if self.args.inpaint:
            criteria_list += ['inpaint_l1', 'inpaint_psnr', 'inpaint_ssim', 'inpaint_vgg']
        if self.args.rank == 0:
            for i in range(self.args.num_pred_step):
                for j in range(self.args.num_pred_once):
                    prefix = 'step_{}_frame_{}_'.format(i,j)
                    for crit in criteria_list:
                        val_criteria[prefix+crit] = AverageMeter()

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

                if self.step % 3 == 0 and self.args.rank == 0:  # visualize
                    vis_coarse_imgs = []
                    vis_inpaint_imgs = []
                    vis_inpaint_masks = []
                    vis_coarse_segs = []

                last_rgb_output = torch.cat([data['frame1'], data['frame2']], dim=1).cuda(self.args.rank, non_blocking=True)
                if self.args.mode == 'xs2xs':
                    last_seg_output = torch.cat([data['seg1'], data['seg2']], dim=1).cuda(self.args.rank, non_blocking=True)
                if self.args.num_pred_step > 1:
                    assert self.args.num_pred_once == 1
                for i in range(self.args.num_pred_step):
                    # 1. update input
                    gt_start_frame_ind = 3+i*self.args.num_pred_once
                    gt_x = torch.cat([data['frame'+str(i)] 
                                         for i in range(gt_start_frame_ind, gt_start_frame_ind+self.args.num_pred_once)], dim=1)\
                                                .cuda(self.args.rank, non_blocking=True)
                    gt_seg =  torch.cat([data['seg'+str(i)] 
                                         for i in range(gt_start_frame_ind, gt_start_frame_ind+self.args.num_pred_once)], dim=1)\
                                                .cuda(self.args.rank, non_blocking=True)
                    x = last_rgb_output
                    seg = last_seg_output
                    if self.args.fix_init_frames:
                        x = torch.cat([data['frame2'].detach().cuda(self.args.rank, non_blocking=True), x], dim=1)
                        seg = torch.cat([data['seg2'].detach().cuda(self.args.rank, non_blocking=True), seg], dim=1)

                    # 2. model training
                    if self.args.inpaint:
                        coarse_img, coarse_seg, inpaint_mask, inpaint_img = self.model(x, seg=seg, gt_x=gt_x, gt_seg=gt_seg)
                    else:
                        coarse_img, coarse_seg = self.model(x, seg=seg, gt_x=gt_x, gt_seg=gt_seg)

                    # 3. update outputs and store them
                    for j in range(self.args.num_pred_once):
                        prefix = 'step_{}_frame_{}'.format(i,j)
                        # rgb criteria
                        step_losses[prefix+'_coarse_l1']    = self.L1Loss(  self.normalize(coarse_img[:,j*3:j*3+3]), 
                                                                            self.normalize(gt_x[:,j*3:j*3+3]))
                        step_losses[prefix+'_coarse_psnr']  = self.PSNRLoss(self.normalize(coarse_img[:,j*3:j*3+3]), 
                                                                            self.normalize(gt_x[:,j*3:j*3+3]))
                        step_losses[prefix+'_coarse_ssim']  = 1-self.SSIMLoss(  self.normalize(coarse_img[:,j*3:j*3+3]), 
                                                                                self.normalize(gt_x[:,j*3:j*3+3]))
                        step_losses[prefix+'_coarse_iou']   =  self.IoULoss(torch.argmax(coarse_seg[:, j*20:j*20+20], dim=1), 
                                                                            torch.argmax(gt_seg[:,j*20:j*20+20], dim=1))
                        step_losses[prefix+'_coarse_vgg']   =  self.VGGCosLoss( self.normalize(coarse_img[:,j*3:j*3+3]), 
                                                                                self.normalize(gt_x[:,j*3:j*3+3]), False)  
                        if self.args.inpaint:
                            step_losses[prefix+'_inpaint_l1']    = self.L1Loss(  self.normalize(inpaint_img[:,j*3:j*3+3]), 
                                                                                self.normalize(gt_x[:,j*3:j*3+3]))
                            step_losses[prefix+'_inpaint_psnr']  = self.PSNRLoss(self.normalize(inpaint_img[:,j*3:j*3+3]), 
                                                                                self.normalize(gt_x[:,j*3:j*3+3]))
                            step_losses[prefix+'_inpaint_ssim']  = 1-self.SSIMLoss(  self.normalize(inpaint_img[:,j*3:j*3+3]), 
                                                                                    self.normalize(gt_x[:,j*3:j*3+3]))
                            step_losses[prefix+'_inpaint_vgg']   =  self.VGGCosLoss( self.normalize(inpaint_img[:,j*3:j*3+3]), 
                                                                                    self.normalize(gt_x[:,j*3:j*3+3]), False)   

                        if self.step % 3 == 0 and self.args.rank == 0: 
                            vis_coarse_imgs.append(coarse_img[0,j*3:j*3+3].cpu())
                            vis_coarse_segs.append(coarse_seg[0,j*20:j*20+20].cpu())

                            if self.args.inpaint:
                                vis_inpaint_masks.append(inpaint_mask[0,j:j+1].cpu())
                                vis_inpaint_imgs.append(inpaint_img[0,j*3:j*3+3].cpu())

                    if self.args.num_pred_step == 1:
                        break
                    back_img = inpainted_img if self.args.inpaint else coarse_img
                    last_rgb_output = torch.cat( [ x[:,-3:], back_img ], dim=1) 
                    last_seg_output = torch.cat( [ seg[:,-20:], 
                                                    torch.eye(20)[coarse_seg.argmax(dim=1)].permute(0,3,1,2).contiguous().cuda(self.args.rank, non_blocking=True)], dim=1)


                self.sync(step_losses) # sum

                comp_time += time() - end
                end = time()

                # print
                if self.args.rank == 0:
                    for i in range(self.args.num_pred_step):
                        for j in range(self.args.num_pred_once):
                            prefix = 'step_{}_frame_{}_'.format(i,j)
                            for crit in criteria_list:
                                key = prefix+crit
                                val_criteria[key].update(step_losses[key].cpu().item(), batch_size*self.args.gpus)

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
                        image_set = self.prepare_image_set(data, vis_coarse_imgs, vis_coarse_segs, vis_inpaint_masks, vis_inpaint_imgs)
                        image_name = 'e{}_img_{}'.format(self.epoch, self.step)
                        self.writer.add_image(image_name, image_set, self.step)

        if self.args.rank == 0:
            log_main = '######################### Epoch [{epoch:d}] Evaluation Results #########################'.format(epoch=self.epoch)

            for i in range(self.args.num_pred_step):
                for j in range(self.args.num_pred_once):
                    prefix = 'step_{}_frame_{}_'.format(i,j)
                    log = '\n\tstep{step:.0f} frame{ind:.0f} coarse l1 [{l1:.3f}] vgg [{vgg:.3f}] psnr [{psnr:.3f}] ssim [{ssim:.3f}] iou [{iou:.3f}]'.format(
                            step=1+i,
                            ind=1+j, 
                            l1=val_criteria[prefix+'coarse_l1'].avg,
                            vgg=val_criteria[prefix+'coarse_vgg'].avg,
                            psnr=val_criteria[prefix+'coarse_psnr'].avg,
                            ssim=val_criteria[prefix+'coarse_ssim'].avg,
                            iou=val_criteria[prefix+'coarse_iou'].avg
                        )
                    log_main+=log
                    if self.args.inpaint:
                        log = '\n\t\t\t inpaint l1 [{l1:.3f}] vgg [{vgg:.3f}] psnr [{psnr:.3f}] ssim [{ssim:.3f}]'.format(
                                l1=val_criteria[prefix+'inpaint_l1'].avg,
                                vgg=val_criteria[prefix+'inpaint_vgg'].avg,
                                psnr=val_criteria[prefix+'inpaint_psnr'].avg,
                                ssim=val_criteria[prefix+'inpaint_ssim'].avg
                            )
                        log_main+=log

            log_main += '\n#####################################################################################\n'

            self.args.logger.info(
                log_main
            )

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
                if self.args.inpaint:
                    coarse_img, seg, inpaint_mask, img = self.model(input_imgs, input_segs)

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

                # print
                # self.args.logger.info(
                #     'pred step [{}] load [{load_time:.3f}s] comp [{comp_time:.3f}s]'.format(self.step+1,
                #         load_time=load_time, comp_time=comp_time
                #     )
                # )
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
        if self.args.inpaint:
            save_dict['inpaint_model'] = self.model.module.inpaint_model.state_dict()
            save_dict['inpaint_opt'] = self.inpaint_opt.state_dict()
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
        if self.args.load_inpaint:
            new_ckpt = OrderedDict()
            assert self.args.inpaint
            inpaint_model_dict = self.model.module.inpaint_model.state_dict()
            for key,item in ckpt['inpaint_model'].items():
                new_ckpt[key] = item
            inpaint_model_dict.update(new_ckpt)
            self.model.module.inpaint_model.load_state_dict(inpaint_model_dict)
        

        # load opt
        if self.args.split == 'train':
            # load coarse opt
            if self.args.train_coarse and self.args.load_coarse:
                self.coarse_opt.load_state_dict(ckpt['coarse_opt'])
                for state in self.coarse_opt.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda(self.args.rank)
            # load inpaint opt
            if self.args.train_inpaint and self.args.load_inpaint:
                assert self.args.inpaint
                self.inpaint_opt.load_state_dict(ckpt['inpaint_opt'])
                for state in self.inpaint_opt.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda(self.args.rank)
        if self.args.resume or self.args.split != 'train':
            assert ckpt['epoch']-1 == self.args.checkepoch, [ckpt['epoch'], self.args.checkepoch]
            self.epoch = ckpt['epoch']
        self.args.logger.info('checkpoint loaded')

