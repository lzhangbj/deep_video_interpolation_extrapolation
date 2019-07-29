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
from losses import RGBLoss, PSNR, SSIM, IoU, GANLoss, VGGCosineLoss, KLDLoss, GANScalarLoss
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


class InterGANTrainer:
    def __init__(self, args):
        self.args = args

        args.logger.info('Initializing trainer')
        self.model = get_model(args)
        if not self.args.train_coarse:
            self.set_net_grad(self.model.coarse_model, False)
        if self.args.frame_disc and not self.args.train_frame_disc: 
            self.set_net_grad(self.model.frame_disc_model, False)
        if self.args.video_disc and not self.args.train_video_disc: 
            self.set_net_grad(self.model.video_disc_model, False)

        params_cnt = count_parameters(self.model.coarse_model)
        args.logger.info("coarse params "+str(params_cnt))
        if self.args.frame_disc:
            params_cnt = count_parameters(self.model.frame_disc_model)
            args.logger.info("frame disc params "+str(params_cnt))
        if self.args.video_disc:
            params_cnt = count_parameters(self.model.video_disc_model)
            args.logger.info("video disc params "+str(params_cnt))
        if self.args.frame_det_disc:
            params_cnt = count_parameters(self.model.frame_det_disc_model)
            args.logger.info("frame det disc params "+str(params_cnt))
        if self.args.video_det_disc:
            params_cnt = count_parameters(self.model.video_det_disc_model)
            args.logger.info("video det disc params "+str(params_cnt))

        torch.cuda.set_device(args.rank)
        self.model.cuda(args.rank)
        self.model = torch.nn.parallel.DistributedDataParallel(self.model,
                device_ids=[args.rank])
        
        if self.args.split in ['train', 'val']:
            train_dataset, val_dataset = get_dataset(args)

        if args.split == 'train':
            # train loss
            self.RGBLoss = RGBLoss(args).cuda(args.rank)
            if self.args.vae:
                self.KLDLoss = KLDLoss(args).cuda(args.rank)
            if self.args.frame_disc:
                self.FrameDisc_DLoss = GANScalarLoss(weight=self.args.frame_disc_disc_weight).cuda(args.rank)
                self.FrameDisc_GLoss = GANScalarLoss(weight=self.args.frame_disc_gen_weight).cuda(args.rank)
            if self.args.video_disc:
                self.VideoDisc_DLoss = GANScalarLoss(weight=self.args.video_disc_disc_weight).cuda(args.rank)
                self.VideoDisc_GLoss = GANScalarLoss(weight=self.args.video_disc_gen_weight).cuda(args.rank)
            if self.args.frame_det_disc:
                self.FrameDetDisc_DLoss = GANScalarLoss(weight=self.args.frame_det_disc_disc_weight).cuda(args.rank)
                self.FrameDetDisc_GLoss = GANScalarLoss(weight=self.args.frame_det_disc_gen_weight).cuda(args.rank)
            if self.args.video_det_disc:
                self.VideoDetDisc_DLoss = GANScalarLoss(weight=self.args.video_det_disc_disc_weight).cuda(args.rank)
                self.VideoDetDisc_GLoss = GANScalarLoss(weight=self.args.video_det_disc_gen_weight).cuda(args.rank)
            self.SegLoss = nn.CrossEntropyLoss().cuda(args.rank)

            self.coarse_opt = torch.optim.Adamax(list(self.model.module.coarse_model.parameters()), lr=args.coarse_learning_rate)
            if self.args.frame_disc:
                self.frame_disc_opt = torch.optim.Adam(list(self.model.module.frame_disc_model.parameters()), lr=args.frame_disc_learning_rate)
            if self.args.video_disc:
                self.video_disc_opt = torch.optim.Adam(list(self.model.module.video_disc_model.parameters()), lr=args.video_disc_learning_rate)
            if self.args.frame_det_disc:
                self.frame_det_disc_opt = torch.optim.Adam(list(self.model.module.frame_det_disc_model.parameters()), lr=args.frame_det_disc_learning_rate)
            if self.args.video_det_disc:
                self.video_det_disc_opt = torch.optim.Adam(list(self.model.module.video_det_disc_model.parameters()), lr=args.video_det_disc_learning_rate)
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


    def set_net_grad(self, net, flag=True):
        for p in net.parameters():
            p.requires_grad = flag

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

    def create_heatmap(self, prob_map):
        c, h, w = prob_map.size()
        assert c==1, c
        rgb_prob_map = torch.zeros(3, h, w)
        minimum, maximum = 0.0, 1.0
        ratio = 2 * (prob_map-minimum) / (maximum - minimum)

        rgb_prob_map[0] = 1-ratio
        rgb_prob_map[1] = ratio-1
        rgb_prob_map[:2].clamp_(0,1)
        rgb_prob_map[2] = 1-rgb_prob_map[0]-rgb_prob_map[1]
        return rgb_prob_map

    # def create_stand_heatmap(self):
    #     heatmap = torch.zeros(3, 128, 128)
    #     for i in range(128):
    #         heatmap[0, :, i] = max(0, 1 - 2.*i/128)
    #         heatmap[1, :, i] = max(0, 2.*i/128 - 1)
    #         heatmap[2, :, i] = 1-heatmap[0, :, i]-heatmap[1, :, i]
    #     return heatmap

    def draw_bbox(self, img, bboxes):
        '''
            img (c, h, w)
            bboxes(4,4) 4 objects only (y1, x1, y2, x2)
        '''
        img_np = img.permute(1,2,0).contiguous().numpy()
        colors = [  (32,32,240),# red
                    (240,32,53),# blue
                    (74,240,32),# green
                    (32,157,240) # orange
                    ]
        for ind, color in enumerate(colors):
            bbox = bboxes[ind]
            cv2.rectangle(img_np, (bbox[1], bbox[0]), (bbox[3], bbox[2]), color, 2)
        return torch.tensor(img_np).permute(2,0,1).contiguous()


    def prepare_image_set(self, data, coarse_img, coarse_seg, real_frame=None, fake_frame=None, real_video=None, fake_video=None,
                            real_frame_det=None, fake_frame_det=None, real_video_det=None, fake_video_det=None):
        '''
            input unnormalized img and seg cpu 
        '''
        # assert len(imgs) == self.args.num_pred_step*self.args.num_pred_once
        num_pred_imgs = self.args.num_pred_step*self.args.num_pred_once if self.args.syn_type == 'extra' else 3
        view_gt_rgbs = [ self.normalize(data['frame'+str(i+1)][0].cpu().detach()) for i in range(3)]
        view_gt_segs = [ vis_seg_mask(data['seg'+str(i+1)][0].unsqueeze(0), 20).squeeze(0) for i in range(3)]

        black_img = torch.zeros_like(view_gt_rgbs[0])

        n_rows = 4
        view_gt_rgbs.insert(2, self.normalize(coarse_img))
        view_gt_segs.insert(2, vis_seg_mask(coarse_seg.unsqueeze(0), 20).squeeze(0))

        view_imgs = view_gt_rgbs + view_gt_segs
        
        if self.args.split =='train':
            if self.args.local_disc:
                real_frame = self.normalize(real_frame)
                fake_frame = self.normalize(fake_frame)
                real_video = self.normalize(real_video)
                fake_video = self.normalize(fake_video)
                view_heatmaps = [
                    self.create_heatmap(real_frame), 
                    self.create_heatmap(fake_frame), 
                    self.create_heatmap(real_video), 
                    self.create_heatmap(fake_video)
                ]
                view_imgs += view_heatmaps
            if 'Det' in self.args.frame_disc_model or 'Det' in self.args.video_disc_model or self.args.frame_det_disc or self.args.video_det_disc:
                bboxes = data['bboxes'][0].cpu().numpy()
                view_bboxes = [
                    self.draw_bbox(view_gt_rgbs[0].cpu().detach(), bboxes[0]),
                    self.draw_bbox(view_gt_rgbs[1].cpu().detach(), bboxes[1]),
                    self.draw_bbox(view_gt_rgbs[2].cpu().detach(), bboxes[1]),
                    self.draw_bbox(view_gt_rgbs[3].cpu().detach(), bboxes[2])
                ]
                view_imgs = view_imgs[:4] + view_bboxes + view_imgs[4:]

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
        if self.args.vae:
            d['coarse_kld_loss_record']=0
        if self.args.frame_disc:
            d['coarse_frame_loss_record']=0
            d['disc_frame_fake_loss_record']=0
            d['disc_frame_real_loss_record']=0
            d['disc_frame_all_loss_record']=0
        if self.args.video_disc:
            d['coarse_video_loss_record']=0
            d['disc_video_fake_loss_record']=0
            d['disc_video_real_loss_record']=0
            d['disc_video_all_loss_record']=0
        if self.args.frame_det_disc:
            d['coarse_frame_det_loss_record']=0
            d['disc_frame_det_fake_loss_record']=0
            d['disc_frame_det_real_loss_record']=0
            d['disc_frame_det_all_loss_record']=0
        if self.args.video_disc:
            d['coarse_video_det_loss_record']=0
            d['disc_video_det_fake_loss_record']=0
            d['disc_video_det_real_loss_record']=0
            d['disc_video_det_all_loss_record']=0
        D.update(d)

        return D

    def update_loss_record_dict(self, record_dict, loss_dict, batch_size):
        record_dict['data_cnt']+=batch_size
        loss_name_list = ['l1', 'gdl', 'ssim', 'vgg']
        if self.args.mode == 'xs2xs':
            loss_name_list.append('ce')
        if self.args.vae:
            loss_name_list.append('kld')
        if self.args.frame_disc:
            loss_name_list.append('frame')
        if self.args.video_disc:
            loss_name_list.append('video')
        if self.args.frame_det_disc:
            loss_name_list.append('frame_det')
        if self.args.video_det_disc:
            loss_name_list.append('video_det')

        for loss_name in loss_name_list:
            record_dict['coarse_{}_loss_record'.format(loss_name)] += \
                            batch_size*loss_dict['coarse_{}_loss'.format(loss_name)].item()
            record_dict['coarse_all_loss_record'] += batch_size*loss_dict['coarse_{}_loss'.format(loss_name)].item()

        disc_name_list = ['frame', 'video']
        if self.args.frame_det_disc:
            disc_name_list.append('frame_det')
        if self.args.video_det_disc:
            disc_name_list.append('video_det')
        for disc_name in disc_name_list:
            for loss_name in ['real', 'fake']:
                record_dict['disc_{}_{}_loss_record'.format(disc_name, loss_name)] += \
                                batch_size*loss_dict['disc_{}_{}_loss'.format(disc_name, loss_name)].item()
                record_dict['disc_{}_all_loss_record'.format(disc_name)] += batch_size*loss_dict['disc_{}_{}_loss'.format(disc_name, loss_name)].item()

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
        
        GAN_TRAIN_STEP = 0

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
            bboxes = data['bboxes'].cuda(self.args.rank,non_blocking=True)
            # 2. model training
            if self.args.mode == 'xs2xs':
                coarse_img, coarse_seg, \
                mu, logvar, \
                D_fake_frame_prob, D_real_frame_prob, \
                D_fake_video_prob, D_real_video_prob, \
                G_fake_frame_prob, G_fake_video_prob, \
                D_fake_frame_det_prob, D_real_frame_det_prob, \
                D_fake_video_det_prob, D_sync_fake_video_det_prob, D_real_video_det_prob, \
                G_fake_frame_det_prob, G_fake_video_det_prob = self.model(x, seg, gt_x, gt_seg, bboxes=bboxes)
            else: # discarded
                coarse_img, mu, logvar, \
                D_fake_frame_prob, D_real_frame_prob, \
                D_fake_video_prob, D_real_video_prob, \
                G_fake_frame_prob, G_fake_video_prob = self.model(x, gt_x=gt_x, bboxes=bboxes)

            # 3. update outputs and store them
            prefix = 'coarse'
            loss_dict.update(self.RGBLoss(self.normalize(coarse_img), self.normalize(gt_x), False, prefix=prefix))
            if self.args.mode == 'xs2xs':
                loss_dict[prefix+'_ce_loss'] = self.args.ce_weight*self.SegLoss(coarse_seg, torch.argmax(gt_seg, dim=1))  
            if self.args.vae:
                loss_dict[prefix+'_kld_loss'] = self.KLDLoss(mu, logvar)
            if self.args.frame_disc:
                loss_dict['coarse_frame_loss']    = self.FrameDisc_GLoss(G_fake_frame_prob, True) if self.global_step > GAN_TRAIN_STEP else \
                                                        self.FrameDisc_GLoss(G_fake_frame_prob, True)*0
                loss_dict['disc_frame_real_loss'] = self.FrameDisc_DLoss(D_real_frame_prob, True) if self.global_step > GAN_TRAIN_STEP else \
                                                        self.FrameDisc_DLoss(D_real_frame_prob, True)*0
                loss_dict['disc_frame_fake_loss'] = self.FrameDisc_DLoss(D_fake_frame_prob, False) if self.global_step > GAN_TRAIN_STEP else \
                                                        self.FrameDisc_DLoss(D_fake_frame_prob, False)*0
            if self.args.video_disc:
                loss_dict['coarse_video_loss']    = self.VideoDisc_GLoss(G_fake_video_prob, True) if self.global_step > GAN_TRAIN_STEP else \
                                                        self.VideoDisc_GLoss(G_fake_video_prob, True)*0
                loss_dict['disc_video_real_loss'] = self.VideoDisc_DLoss(D_real_video_prob, True) if self.global_step > GAN_TRAIN_STEP else \
                                                        self.VideoDisc_DLoss(D_real_video_prob, True)*0
                loss_dict['disc_video_fake_loss'] = self.VideoDisc_DLoss(D_fake_video_prob, False) if self.global_step > GAN_TRAIN_STEP else \
                                                        self.VideoDisc_DLoss(D_fake_video_prob, False)*0
            if self.args.frame_det_disc:
                loss_dict['coarse_frame_det_loss']    = self.FrameDetDisc_GLoss(G_fake_frame_det_prob, True) if self.global_step > GAN_TRAIN_STEP else \
                                                        self.FrameDetDisc_GLoss(G_fake_frame_det_prob, True)*0
                loss_dict['disc_frame_det_real_loss'] = self.FrameDetDisc_DLoss(D_real_frame_det_prob, True) if self.global_step > GAN_TRAIN_STEP else \
                                                        self.FrameDetDisc_DLoss(D_real_frame_det_prob, True)*0
                loss_dict['disc_frame_det_fake_loss'] = self.FrameDetDisc_DLoss(D_fake_frame_det_prob, False) if self.global_step > GAN_TRAIN_STEP else \
                                                        self.FrameDetDisc_DLoss(D_fake_frame_det_prob, False)*0
            if self.args.video_det_disc:
                loss_dict['coarse_video_det_loss']    = self.VideoDetDisc_GLoss(G_fake_video_det_prob, True) if self.global_step > GAN_TRAIN_STEP else \
                                                        self.VideoDetDisc_GLoss(G_fake_video_det_prob, True)*0
                loss_dict['disc_video_det_real_loss'] = self.VideoDetDisc_DLoss(D_real_video_det_prob, True) if self.global_step > GAN_TRAIN_STEP else \
                                                        self.VideoDetDisc_DLoss(D_real_video_det_prob, True)*0
                loss_dict['disc_video_det_fake_loss'] = self.VideoDetDisc_DLoss((D_fake_video_det_prob+D_sync_fake_video_det_prob)/2, False) if self.global_step > GAN_TRAIN_STEP else \
                                                        self.VideoDetDisc_DLoss((D_fake_video_det_prob+D_sync_fake_video_det_prob)/2, False)*0

            loss = 0
            for i in loss_dict.values():
                loss += torch.mean(i)
            loss_dict['loss_all'] = loss
            self.sync(loss_dict)
            # backward pass
            self.coarse_opt.zero_grad()     if self.args.train_coarse  else None
            self.frame_disc_opt.zero_grad() if self.args.train_frame_disc  else None
            self.video_disc_opt.zero_grad() if self.args.train_video_disc  else None
            self.frame_det_disc_opt.zero_grad() if self.args.train_frame_det_disc  else None
            self.video_det_disc_opt.zero_grad() if self.args.train_video_det_disc  else None
            loss_dict['loss_all'].backward()
            self.coarse_opt.step()  if self.args.train_coarse  else None
            self.frame_disc_opt.step()  if self.args.train_frame_disc and  self.global_step > GAN_TRAIN_STEP else None
            self.video_disc_opt.step()  if self.args.train_video_disc and  self.global_step > GAN_TRAIN_STEP else None
            self.frame_det_disc_opt.step()  if self.args.train_frame_det_disc and  self.global_step > GAN_TRAIN_STEP else None
            self.video_det_disc_opt.step()  if self.args.train_video_det_disc and  self.global_step > GAN_TRAIN_STEP else None
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
                    if self.args.vae:
                        log+= ' kld [{kld:.3f}]'.format(kld=step_loss_record_dict['coarse_kld_loss_record'])
                    if self.args.frame_disc:
                        log+= ' frame [{frame:.3f}]'.format(frame=step_loss_record_dict['coarse_frame_loss_record'])
                    if self.args.video_disc:
                        log+= ' video [{video:.3f}]'.format(video=step_loss_record_dict['coarse_video_loss_record'])
                    if self.args.frame_det_disc:
                        log+= ' frame det [{frame:.3f}]'.format(frame=step_loss_record_dict['coarse_frame_det_loss_record'])
                    if self.args.video_det_disc:
                        log+= ' video det [{video:.3f}]'.format(video=step_loss_record_dict['coarse_video_det_loss_record'])
                    log+= ' all [{all:.3f}]'.format(all=step_loss_record_dict['coarse_all_loss_record'])
                    if self.args.frame_disc:
                        log+= '\n\t\t\tframe disc real [{f_r:.3f}] fake [{f_f:.3f}]'.format(
                                            f_r=step_loss_record_dict['disc_frame_real_loss_record'],
                                            f_f=step_loss_record_dict['disc_frame_fake_loss_record']
                                        )
                    if self.args.video_disc:
                        log+= '\n\t\t\tvideo disc real [{f_r:.3f}] fake [{f_f:.3f}]'.format(
                                            f_r=step_loss_record_dict['disc_video_real_loss_record'],
                                            f_f=step_loss_record_dict['disc_video_fake_loss_record']
                                        )
                    if self.args.frame_det_disc:
                        log+= '\n\t\t\tframe det disc real [{f_r:.3f}] fake [{f_f:.3f}]'.format(
                                            f_r=step_loss_record_dict['disc_frame_det_real_loss_record'],
                                            f_f=step_loss_record_dict['disc_frame_det_fake_loss_record']
                                        )
                    if self.args.video_det_disc:
                        log+= '\n\t\t\tvideo det disc real [{f_r:.3f}] fake [{f_f:.3f}]'.format(
                                            f_r=step_loss_record_dict['disc_video_det_real_loss_record'],
                                            f_f=step_loss_record_dict['disc_video_det_fake_loss_record']
                                        )
                    log_main+=log
                    log_main += '\n\t\t\tloss total [{:.3f}]'.format(step_loss_record_dict['all_loss_record'])

                    self.args.logger.info(log_main)
                    # self.args.logger.info('hhhhhhhhhhhh')
                    comp_time = 0
                    load_time = 0

                    if step_loss_record_dict['data_cnt'] != 0:
                        for key, value in step_loss_record_dict.items():
                            step_loss_record_dict[key] = 0

                if self.step % 30 == 0: 
                    image_set = self.prepare_image_set(data, coarse_img[0].clamp_(-1,1).cpu(), coarse_seg[0].cpu(),
                                                            D_real_frame_prob[0].clamp_(-1,1).cpu(), D_fake_frame_prob[0].clamp_(-1,1).cpu(),
                                                            D_real_video_prob[0].clamp_(-1,1).cpu(), D_fake_video_prob[0].clamp_(-1,1).cpu(),
                                                            D_real_frame_det_prob[0].clamp_(-1,1).cpu() if D_real_frame_det_prob is not None else None, 
                                                            D_fake_frame_det_prob[0].clamp_(-1,1).cpu() if D_fake_frame_det_prob is not None else None,
                                                            D_real_video_det_prob[0].clamp_(-1,1).cpu() if D_real_video_det_prob is not None else None, 
                                                            D_fake_video_det_prob[0].clamp_(-1,1).cpu() if D_fake_video_det_prob is not None else None
                                                        )
                    img_name = 'image_{} frame real {:.3f} fake {:.3f} video real {:.3f} fake {:.3f} '.format(
                                                self.global_step, 
                                                D_real_frame_prob[0].mean().item(), D_fake_frame_prob[0].mean().item(), 
                                                D_real_video_prob[0].mean().item(), D_fake_video_prob[0].mean().item())
                    if self.args.frame_det_disc:
                        img_name+= ' frame det real {:.3f} fake {:.3f}'.format(D_real_frame_det_prob[0].mean().item(), D_fake_frame_det_prob[0].mean().item())
                    if self.args.video_det_disc:
                        img_name+= ' video det real {:.3f} fake {:.3f} sync_fake {:.3f}'\
                                                .format(D_real_video_det_prob[0].mean().item(), 
                                                        D_fake_video_det_prob[0].mean().item(),
                                                        D_sync_fake_video_det_prob[0].mean().item())
                    self.writer.add_image(img_name, image_set, self.global_step)

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
            if self.args.vae:
                log+= ' kld [{kld:.3f}]'.format(kld=epoch_loss_record_dict['coarse_kld_loss_record'])
            if self.args.frame_disc:
                log+= ' frame [{frame:.3f}]'.format(frame=epoch_loss_record_dict['coarse_frame_loss_record'])
            if self.args.video_disc:
                log+= ' video [{video:.3f}]'.format(video=epoch_loss_record_dict['coarse_video_loss_record'])
            if self.args.frame_det_disc:
                log+= ' frame det [{frame:.3f}]'.format(frame=epoch_loss_record_dict['coarse_frame_det_loss_record'])
            if self.args.video_det_disc:
                log+= ' video det [{video:.3f}]'.format(video=epoch_loss_record_dict['coarse_video_det_loss_record'])
            log+= ' all [{all:.3f}]'.format(all=epoch_loss_record_dict['coarse_all_loss_record'])
            if self.args.frame_disc:
                log+= '\n\t\t\tframe disc real [{f_r:.3f}] fake [{f_f:.3f}]'.format(
                                    f_r=epoch_loss_record_dict['disc_frame_real_loss_record'],
                                    f_f=epoch_loss_record_dict['disc_frame_fake_loss_record']
                                )
            if self.args.video_disc:
                log+= '\n\t\t\tvideo disc real [{f_r:.3f}] fake [{f_f:.3f}]'.format(
                                    f_r=epoch_loss_record_dict['disc_video_real_loss_record'],
                                    f_f=epoch_loss_record_dict['disc_video_fake_loss_record']
                                )
            if self.args.frame_det_disc:
                log+= '\n\t\t\tframe det disc real [{f_r:.3f}] fake [{f_f:.3f}]'.format(
                                    f_r=epoch_loss_record_dict['disc_frame_det_real_loss_record'],
                                    f_f=epoch_loss_record_dict['disc_frame_det_fake_loss_record']
                                )
            if self.args.video_det_disc:
                log+= '\n\t\t\tvideo det disc real [{f_r:.3f}] fake [{f_f:.3f}]'.format(
                                    f_r=epoch_loss_record_dict['disc_video_det_real_loss_record'],
                                    f_f=epoch_loss_record_dict['disc_video_det_fake_loss_record']
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
        criteria_list = ['l1', 'psnr', 'ssim', 'vgg']
        if self.args.mode == 'xs2xs':
            criteria_list.append('iou')
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
                bboxes = data['bboxes'].cuda(self.args.rank,non_blocking=True)
                # 2. model training
                if self.args.mode == 'xs2xs':
                    coarse_img, coarse_seg, \
                    mu, logvar, \
                    D_fake_frame_prob, D_real_frame_prob, \
                    D_fake_video_prob, D_real_video_prob, \
                    G_fake_frame_prob, G_fake_video_prob, \
                    D_fake_frame_det_prob, D_real_frame_det_prob, \
                    D_fake_video_det_prob, D_real_video_det_prob, \
                    G_fake_frame_det_prob, G_fake_video_det_prob = self.model(x, seg, gt_x, gt_seg, bboxes=bboxes)
                else: # discarded
                    coarse_img, mu, logvar, \
                    D_fake_frame_prob, D_real_frame_prob, \
                    D_fake_video_prob, D_real_video_prob, \
                    G_fake_frame_prob, G_fake_video_prob = self.model(x, gt_x=gt_x, bboxes=bboxes)

                # 3. update outputs and store them
                # rgb criteria
                step_losses['l1']    = self.L1Loss(  self.normalize(coarse_img), 
                                                                    self.normalize(gt_x))
                step_losses['psnr']  = self.PSNRLoss(self.normalize(coarse_img), 
                                                                    self.normalize(gt_x))
                step_losses['ssim']  = 1-self.SSIMLoss(  self.normalize(coarse_img), 
                                                                        self.normalize(gt_x))
                step_losses['iou']   =  self.IoULoss(torch.argmax(coarse_seg, dim=1), 
                                                                    torch.argmax(gt_seg, dim=1))
                step_losses['vgg']   =  self.VGGCosLoss( self.normalize(coarse_img), 
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
                        # image_set = self.prepare_image_set(data, coarse_img[0].clamp_(-1,1).cpu(), coarse_seg[0].cpu(),
                        #                                     D_real_frame_prob[0].clamp_(-1,1).cpu(), D_fake_frame_prob[0].clamp_(-1,1).cpu(),
                        #                                     D_real_video_prob[0].clamp_(-1,1).cpu(), D_fake_video_prob[0].clamp_(-1,1).cpu()
                        #                                 )
                        image_set = self.prepare_image_set(data, coarse_img[0].clamp_(-1,1).cpu(), coarse_seg[0].cpu())
                        img_name = 'e{}_img_{}'.format(
                                                self.epoch,
                                                self.step)
                        self.writer.add_image(img_name, image_set, self.step)

        if self.args.rank == 0:
            log_main = '\n######################### Epoch [{epoch:d}] Evaluation Results #########################'.format(epoch=self.epoch)

            log = '\n\tcoarse l1 [{l1:.3f}] vgg [{vgg:.3f}] psnr [{psnr:.3f}] ssim [{ssim:.3f}] iou [{iou:.3f}]'.format(
                    l1=val_criteria['l1'].avg,
                    vgg=val_criteria['vgg'].avg,
                    psnr=val_criteria['psnr'].avg,
                    ssim=val_criteria['ssim'].avg,
                    iou=val_criteria['iou'].avg
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
            'coarse_model': self.model.module.coarse_model.cpu().state_dict(),
            'coarse_opt': self.coarse_opt.state_dict(),
        }
        if self.args.frame_disc:
            save_dict['frame_disc_model'] = self.model.module.frame_disc_mode.state_dict()
            save_dict['frame_disc_opt'] = self.frame_disc_opt.state_dict()
        if self.args.video_disc:
            save_dict['video_disc_model'] = self.model.module.video_disc_model.state_dict()
            save_dict['video_disc_opt'] = self.video_disc_opt.state_dict()
        if self.args.frame_det_disc:
            save_dict['frame_det_disc_model'] = self.model.module.frame_det_disc_model.state_dict()
            save_dict['frame_det_disc_opt'] = self.frame_det_disc_opt.state_dict()
        if self.args.video_det_disc:
            save_dict['video_det_disc_model'] = self.model.module.video_det_disc_model.state_dict()
            save_dict['video_det_disc_opt'] = self.video_det_disc_opt.state_dict()
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

        if self.args.load_frame_disc:
            new_ckpt = OrderedDict()
            assert self.args.frame_disc
            frame_disc_model_dict = self.model.module.frame_disc_model.state_dict()
            for key,item in ckpt['frame_disc_model'].items():
                new_ckpt[key] = item
            frame_disc_model_dict.update(new_ckpt)
            self.model.module.frame_disc_model.load_state_dict(frame_disc_model_dict)
        
        if self.args.load_video_disc:
            new_ckpt = OrderedDict()
            assert self.args.video_disc
            video_disc_model_dict = self.model.module.video_disc_model.state_dict()
            for key,item in ckpt['video_disc_model'].items():
                new_ckpt[key] = item
            video_disc_model_dict.update(new_ckpt)
            self.model.module.video_disc_model.load_state_dict(video_disc_model_dict)

        if self.args.load_frame_det_disc:
            new_ckpt = OrderedDict()
            assert self.args.frame_det_disc
            frame_det_disc_model_dict = self.model.module.frame_det_disc_model.state_dict()
            for key,item in ckpt['frame_det_disc_model'].items():
                new_ckpt[key] = item
            frame_det_disc_model_dict.update(new_ckpt)
            self.model.module.frame_det_disc_model.load_state_dict(frame_det_disc_model_dict)
        
        if self.args.load_video_det_disc:
            new_ckpt = OrderedDict()
            assert self.args.video_det_disc
            video_det_disc_model_dict = self.model.module.video_det_disc_model.state_dict()
            for key,item in ckpt['video_det_disc_model'].items():
                new_ckpt[key] = item
            video_det_disc_model_dict.update(new_ckpt)
            self.model.module.video_det_disc_model.load_state_dict(video_det_disc_model_dict)
        

        # load opt
        if self.args.split == 'train':
            # load coarse opt
            if self.args.train_coarse and self.args.load_coarse:
                self.coarse_opt.load_state_dict(ckpt['coarse_opt'])
                for state in self.coarse_opt.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda(self.args.rank)
            # load frame opt
            if self.args.train_frame_disc and self.args.load_frame_disc:
                assert self.args.frame_disc
                self.frame_disc_opt.load_state_dict(ckpt['frame_disc_opt'])
                for state in self.frame_disc_opt.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda(self.args.rank)
            # load video_disc opt
            if self.args.train_video_disc and self.args.load_video_disc:
                assert self.args.video_disc
                self.video_disc_opt.load_state_dict(ckpt['video_disc_opt'])
                for state in self.video_disc_opt.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda(self.args.rank)
            # load frame opt
            if self.args.train_frame_det_disc and self.args.load_frame_det_disc:
                assert self.args.frame_det_disc
                self.frame_det_disc_opt.load_state_dict(ckpt['frame_det_disc_opt'])
                for state in self.frame_det_disc_opt.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda(self.args.rank)
            # load video_disc opt
            if self.args.train_video_det_disc and self.args.load_video_det_disc:
                assert self.args.video_det_disc
                self.video_det_disc_opt.load_state_dict(ckpt['video_det_disc_opt'])
                for state in self.video_det_disc_opt.state.values():
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

