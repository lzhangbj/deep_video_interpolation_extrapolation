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
from torchvision.utils import make_grid
# from utils import AverageMeter
# from loss import CombinedLoss
from losses import RGBLoss, PSNR, SSIM, IoU, GANLoss, VGGCosineLoss, losses_multigpu_only_mask
import nets

from data import get_dataset
from utils.net_utils import *
# from cfg import cfg

def get_model(args):
    # build model
    model = nets.__dict__[args.model](args)
    return model


class VAEer:
    def __init__(self, args):
        self.args = args

        args.logger.info('Initializing trainer')
        # if not os.path.isdir('../predict'):       only used in validation
        #     os.makedirs('../predict')
        self.model = get_model(args)
        torch.cuda.set_device(args.rank)
        self.model.cuda(args.rank)
        self.model = torch.nn.parallel.DistributedDataParallel(self.model,
                device_ids=[args.rank])
        train_dataset, val_dataset = get_dataset(args)

        if not args.val:
            # train loss
            # self.RGBLoss = RGBLoss(args, sharp=True)
            self.RGBLoss = losses_multigpu_only_mask(args, self.model.module.floww)
            self.SegLoss = nn.CrossEntropyLoss()
            self.RGBLoss.cuda(args.rank)
            self.SegLoss.cuda(args.rank)

            if args.optimizer == "adamax":
                self.optimizer = torch.optim.Adamax(list(self.model.parameters()), lr=args.learning_rate)
            elif args.optimizer == "adam":
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)
            elif args.optimizer == "sgd":
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.learning_rate, momentum=0.9)

            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            self.train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size//args.gpus, shuffle=False,
                num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)

        else:
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
        if args.resume or (args.val and not args.checkepoch_range):
            self.load_checkpoint()

        if args.rank == 0:
            if args.val:
                self.writer =  SummaryWriter(args.path+'/val_logs') if args.interval == 2 else\
                                SummaryWriter(args.path+'/val_int_1_logs')
            else:
                self.writer = SummaryWriter(args.path+'/logs')


    def set_epoch(self, epoch):
        self.args.logger.info("Start of epoch %d" % (epoch+1))
        self.epoch = epoch + 1
        self.train_loader.sampler.set_epoch(epoch)
        # self.val_loader.sampler.set_epoch(epoch)

    def get_input(self, data):
        # if self.args.mode == 'xs2xs':
        #     if self.args.syn_type == 'extra':
        #         x = torch.cat([data['frame1'], data['frame2'], data['seg1'], data['seg2']], dim=1)
        #         gt = torch.cat([data['frame3'], data['seg3']], dim=1)
        #     else:
        #         x = torch.cat([data['frame1'], data['frame3'], data['seg1'], data['seg3']], dim=1)
        #         gt = torch.cat([data['frame2'], data['seg2']], dim=1)        
        # elif self.args.mode == 'xss2x':
        #     if self.args.syn_type == 'extra':
        #         x = torch.cat([data['frame1'], data['frame2'], data['seg1'], data['seg2'], data['seg3']], dim=1)
        #         gt = data['frame3']   
        #     else:
        #         x = torch.cat([data['frame1'], data['frame3'], data['seg1'], data['seg2'], data['seg3']], dim=1)
        #         gt = data['frame2']   
        # return x, gt   
        return None

    def normalize(self, img):
        return (img+1)/2

    def prepare_image_set(self, data, y_pred, y_pred_before_refine, flow, flowback, mask_fw, mask_bw,\
                            seg_pred=None, seg_gt=None):
        view_gt_rgbs = [   data['frames'][0, i] for i in range(self.args.vid_length+1) ]
        blank_image = torch.zeros_like(view_gt_rgbs[0])
        blank_image_bw = self.create_heatmap(torch.zeros_like(mask_fw[0,0]))

        view_pred_rgbs = [blank_image]
        view_pred_refined_rgbs = [blank_image]
        view_flow = [blank_image]
        view_flowback = [blank_image]
        view_mask = [blank_image_bw]
        view_maskback = [blank_image_bw]

        for i in range(self.args.vid_length):
            view_pred_rgbs.append(y_pred_before_refine[0, i])
            view_pred_refined_rgbs.append(y_pred[0, i])
            view_flow.append(self.vis_flow(flow[0,:,i]))
            view_flowback.append(self.vis_flow(flowback[0,:,i]))
            view_mask.append(self.create_heatmap(mask_fw[0,i]))
            view_maskback.append(self.create_heatmap(mask_bw[0,i]))
        if self.args.seg:
            view_seg_gt = [vis_seg_mask(seg_gt[0, i].cpu().unsqueeze(0), 20).squeeze(0) for i in range(self.args.vid_length+1)]
            view_seg_pred = [view_seg_gt[0]] + [vis_seg_mask(seg_pred[0, i].cpu().unsqueeze(0), 20).squeeze(0) for i in range(self.args.vid_length)]
            write_in_img = make_grid(view_gt_rgbs + view_seg_gt + \
                                view_pred_rgbs + view_pred_refined_rgbs + view_seg_pred +\
                                view_flow + view_flowback +\
                                view_mask + view_maskback , nrow=self.args.vid_length+1)
        else:
            write_in_img = make_grid(view_gt_rgbs + \
                        view_pred_rgbs + view_pred_refined_rgbs + \
                        view_flow + view_flowback +\
                        view_mask + view_maskback , nrow=self.args.vid_length+1)
        return write_in_img

    def vis_flow(self, flow_map):
        '''
            input flow map (2, h, w)
        '''
        flow_map_np = flow_map.data.numpy()*10
        u = flow_map_np[0]
        v = flow_map_np[1]
        image = compute_color(u, v)
        image = np.transpose(image, (2,0,1))
        return torch.tensor(image).float()

    def train(self):
        self.args.logger.info('Training started')
        self.model.train()
        end = time()
        load_time = 0
        comp_time = 0
        for step, data in enumerate(self.train_loader):
            self.step = step
            load_time += time() - end
            end = time()
            # for tensorboard
            self.global_step += 1
            # forward pass
            rgb_data = data['frames'].cuda(self.args.rank, non_blocking=True)
            seg_data = data['segs'].cuda(self.args.rank, non_blocking=True)
            disparity_data = data['disparities'].cuda(self.args.rank, non_blocking=True)
            fg_masks = data['fg_masks'].cuda(self.args.rank, non_blocking=True)
            bg_masks = data['bg_masks'].cuda(self.args.rank, non_blocking=True)

            noise_bg = torch.randn(rgb_data[:, 0].size()).cuda(self.args.rank, non_blocking=True)


            if not self.args.seg and not self.args.disparity:
                y_pred_before_refine, y_pred, mu, logvar, flow, flowback, mask_fw, mask_bw, prediction_vgg_feature, gt_vgg_feature = \
                            self.model(rgb_data, seg_data, bg_masks, fg_masks, noise_bg)

                loss_dict = self.RGBLoss(rgb_data,
                        y_pred, mu, logvar, flow, flowback,
                        mask_fw, mask_bw, prediction_vgg_feature, gt_vgg_feature,
                        y_pred_before_refine=y_pred_before_refine)
            elif self.args.disparity: # seg + image _ disparity
                pass
            else: # only seg + image
                y_pred_before_refine, y_pred, seg_pred_before_refine, seg_pred, mu, logvar, flow, flowback, mask_fw, mask_bw, prediction_vgg_feature, gt_vgg_feature = \
                            self.model(rgb_data, seg_data, bg_masks, fg_masks, noise_bg)

                loss_dict = self.RGBLoss(rgb_data,
                        y_pred, mu, logvar, flow, flowback,
                        mask_fw, mask_bw, prediction_vgg_feature, gt_vgg_feature,
                        y_pred_before_refine=y_pred_before_refine, seg_pred=seg_pred, seg_data=seg_data)                
            # if self.args.mode == 'xs2xs':
            #    loss_dict['ce_loss'] = self.args.ce_weight*self.SegLoss(seg, torch.argmax(gt[:,3:], dim=1))   

            # loss and accuracy
            loss = 0
            for i in loss_dict.values():
                loss += torch.mean(i)
            loss_dict['loss_all'] = loss
            
            self.sync(loss_dict)
            # backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # init step in the first train steps
            self.optimizer.step()
            comp_time += time() - end
            end = time()

            if self.args.rank == 0:
                # add info to tensorboard
                info = {key:value.item() for key,value in loss_dict.items()}
                self.writer.add_scalars("losses", info, self.global_step)
                # print
                if self.step % self.args.disp_interval == 0:
                    self.args.logger.info(
                        'Epoch [{epoch:d}/{tot_epoch:d}][{cur_batch:d}/{tot_batch:d}] '
                        'load [{load_time:.3f}s] comp [{comp_time:.3f}s] '
                        'loss [{loss:.4f}]'.format(
                            epoch=self.epoch, tot_epoch=self.args.epochs,
                            cur_batch=self.step+1, tot_batch=len(self.train_loader),
                            load_time=load_time, comp_time=comp_time,
                            loss=loss.item()
                        )
                    )
                    comp_time = 0
                    load_time = 0
                if self.step % 50 == 0:
                    if not self.args.seg: 
                        image_set = self.prepare_image_set(data, y_pred.cpu(), y_pred_before_refine.cpu(), \
                                                flow.cpu(), flowback.cpu(), mask_fw.cpu(), mask_bw.cpu())
                    else:
                        image_set = self.prepare_image_set(data, y_pred.cpu(), y_pred_before_refine.cpu(),  \
                                                flow.cpu(), flowback.cpu(), mask_fw.cpu(), mask_bw.cpu(),\
                                                seg_pred=seg_pred.cpu(), seg_gt=seg_data)
                    self.writer.add_image('image_{}'.format(self.global_step), image_set, self.global_step)


    def validate(self):
        self.args.logger.info('Validation epoch {} started'.format(self.epoch))
        self.model.eval()

        val_criteria = {
            'l1': AverageMeter(),
            'psnr':AverageMeter(),
            'ssim':AverageMeter(),
            'iou':AverageMeter(),
            'vgg':AverageMeter()
        }
        step_losses = OrderedDict()

        with torch.no_grad():
            end = time()
            load_time = 0
            comp_time = 0
            for i, data in enumerate(self.val_loader):
                load_time += time()-end
                end = time()
                self.step=i

                # forward pass
                x, gt = self.get_input(data)
                size = x.size(0)
                x = x.cuda(self.args.rank, non_blocking=True)
                gt = gt.cuda(self.args.rank, non_blocking=True)
                
                img, seg = self.model(x)

                # rgb criteria
                step_losses['l1'] =   self.L1Loss(img, gt[:,:3])
                step_losses['psnr'] = self.PSNRLoss((img+1)/2, (gt[:,:3]+1)/2)
                step_losses['ssim'] = 1-self.SSIMLoss(img, gt[:,:3])
                step_losses['iou'] =  self.IoULoss(torch.argmax(seg, dim=1), torch.argmax(gt[:,3:], dim=1))
                step_losses['vgg'] =  self.VGGCosLoss(img, gt[:, :3], False)
                self.sync(step_losses) # sum
                for key in list(val_criteria.keys()):
                    val_criteria[key].update(step_losses[key].cpu().item(), size*self.args.gpus)

                if self.args.syn_type == 'extra':
                    imgs = []
                    segs = []
                    img = img[0].unsqueeze(0)
                    seg = seg[0].unsqueeze(0)
                    x = x[0].unsqueeze(0)
                    for i in range(self.args.extra_length):
                        if i!=0:
                            x = torch.cat([x[:,3:6], img, x[:, 26:46], seg_fil], dim=1).cuda(self.args.rank, non_blocking=True)
                            img, seg = self.model(x)
                        seg_fil = torch.argmax(seg, dim=1)
                        seg_fil = transform_seg_one_hot(seg_fil, 20, cuda=True)*2-1
                        imgs.append(img)
                        segs.append(seg_fil)
                        



                comp_time += time() - end
                end = time()

                # print
                if self.args.rank == 0:
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
                        if self.args.syn_type == 'inter':
                            image_set = self.prepare_image_set(data, img, seg)
                        else:
                            image_set = self.prepare_image_set(data, imgs, segs, True)
                        image_name = 'e{}_img_{}'.format(self.epoch, self.step)
                        self.writer.add_image(image_name, image_set, self.step)

        if self.args.rank == 0:
            self.args.logger.info(
'Epoch [{epoch:d}]      \n \
L1\t: {l1:.4f}     \n\
PSNR\t: {psnr:.4f}   \n\
SSIM\t: {ssim:.4f}   \n\
IoU\t: {iou:.4f}    \n\
vgg\t: {vgg:.4f}\n'.format(
                    epoch=self.epoch,
                    l1=val_criteria['l1'].avg,
                    psnr=val_criteria['psnr'].avg,
                    ssim=val_criteria['ssim'].avg,
                    iou=val_criteria['iou'].avg,
                    vgg = val_criteria['vgg'].avg
                )
            )
            tfb_info = {key:value.avg for key,value in val_criteria.items()}
            self.writer.add_scalars('val/score', tfb_info, self.epoch)


    def sync(self, loss_dict, mean=True):
        '''Synchronize all tensors given using mean or sum.'''
        for tensor in loss_dict.values():
            dist.all_reduce(tensor)
            if mean:
                tensor.div_(self.args.gpus)

    def create_heatmap(self, bw_map):
        h, w = bw_map.size()
        assert h==128, h
        rgb_prob_map = torch.zeros(3, h, w)
        minimum, maximum = 0.0, 1.0
        ratio = 2 * (bw_map-minimum) / (maximum - minimum)

        rgb_prob_map[0] = 1-ratio
        rgb_prob_map[1] = ratio-1
        rgb_prob_map[:2].clamp_(0,1)
        rgb_prob_map[2] = 1-rgb_prob_map[0]-rgb_prob_map[1]
        return rgb_prob_map

    def save_checkpoint(self):
        save_md_dir = '{}_{}_{}_{}'.format(self.args.model, self.args.mode, self.args.syn_type, self.args.session)
        save_name = os.path.join(self.args.path, 
                                'checkpoint',
                                save_md_dir + '_{}_{}.pth'.format(self.epoch, self.step))
        self.args.logger.info('Saving checkpoint..')
        torch.save({
            'session': self.args.session,
            'epoch': self.epoch + 1,
            'model': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, save_name)
        self.args.logger.info('save model: {}'.format(save_name))

    def load_checkpoint(self):
        load_md_dir = '{}_{}_{}_{}'.format(self.args.model, self.args.mode, self.args.syn_type, self.args.checksession)
        if self.args.load_dir is not None:
            load_name = os.path.join(self.args.load_dir,
                                    'checkpoint',
                                    load_md_dir+'_{}_{}.pth'.format(self.args.checkepoch, self.args.checkpoint))
        else:
            load_name = os.path.join(load_md_dir+'_{}_{}.pth'.format(self.args.checkepoch, self.args.checkpoint))
        self.args.logger.info('Loading checkpoint %s' % load_name)
        ckpt = torch.load(load_name)
        self.model.module.load_state_dict(ckpt['model'])
        # transfer opt params to current device
        if not self.args.val:
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.epoch = ckpt['epoch']
            self.global_step = (self.epoch-1)*len(self.train_loader)
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda(self.args.rank)
        else:
            assert ckpt['epoch']-1 == self.args.checkepoch, [ckpt['epoch'], self.args.checkepoch]
            self.epoch = ckpt['epoch'] - 1
        self.args.logger.info('checkpoint loaded')

