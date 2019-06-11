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
from losses import RGBLoss, PSNR, SSIM, IoU, GANLoss, VGGCosineLoss
import nets

from data import get_dataset
from utils.net_utils import *
# from cfg import cfg

def get_model(args):
    # build model
    model = nets.__dict__[args.model](args)
    return model


class Trainer:
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
            self.RGBLoss = RGBLoss(args, sharp=True)
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
        if self.args.mode == 'xs2xs':
            if self.args.syn_type == 'extra':
                x = torch.cat([data['frame1'], data['frame2'], data['seg1'], data['seg2']], dim=1)
                mask = torch.cat([data['fg_mask1'],data['fg_mask2']], dim=1)
                gt = torch.cat([data['frame3'], data['seg3']], dim=1)
            else:
                x = torch.cat([data['frame1'], data['frame3'], data['seg1'], data['seg3']], dim=1)
                mask = torch.cat([data['fg_mask1'],data['fg_mask3']], dim=1)
                gt = torch.cat([data['frame2'], data['seg2']], dim=1)        
        elif self.args.mode == 'xss2x':
            if self.args.syn_type == 'extra':
                x = torch.cat([data['frame1'], data['frame2'], data['seg1'], data['seg2'], data['seg3']], dim=1)
                gt = data['frame3']   
            else:
                x = torch.cat([data['frame1'], data['frame3'], data['seg1'], data['seg2'], data['seg3']], dim=1)
                gt = data['frame2']   
        return x, mask, gt   

    def normalize(self, img):
        return (img+1)/2

    def prepare_image_set(self, data, img, seg, extra=False):
        view_rgbs = [   self.normalize(data['frame1'][0]), 
                        self.normalize(data['frame2'][0]), 
                        self.normalize(data['frame3'][0])   ]
        view_segs = [vis_seg_mask(data['seg'+str(i)][0].unsqueeze(0), 20).squeeze(0) for i in range(1, 4)]

        if not extra:
            pred_rgb = self.normalize(img[0].cpu())
            pred_seg = vis_seg_mask(seg[0].cpu().unsqueeze(0), 20).squeeze(0) if self.args.mode == 'xs2xs' else torch.zeros_like(view_segs[0])
            insert_index = 2 if self.args.syn_type == 'inter' else 3
            
            view_rgbs.insert(insert_index, pred_rgb)
            view_segs.insert(insert_index, pred_seg)
            write_in_img = make_grid(view_rgbs + view_segs, nrow=4)
        else:
            view_rgbs.insert(3, torch.zeros_like(view_rgbs[-1]))
            view_segs.insert(3, torch.zeros_like(view_segs[-1]))

            view_pred_rgbs = []
            view_pred_segs = []
            for i in range(self.args.extra_length):
                pred_rgb = self.normalize(img[i][0].cpu())
                pred_seg = vis_seg_mask(seg[i].cpu(), 20).squeeze(0) if self.args.mode == 'xs2xs' else torch.zeros_like(view_segs[0])
                view_pred_rgbs.append(pred_rgb)
                view_pred_segs.append(pred_seg)
            write_in_img = make_grid(view_rgbs + view_segs + view_pred_rgbs + view_pred_segs, nrow=4)

        
        return write_in_img

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
            x, fg_mask, gt = self.get_input(data)
            x = x.cuda(self.args.rank, non_blocking=True)
            fg_mask = fg_mask.cuda(self.args.rank, non_blocking=True)
            gt = gt.cuda(self.args.rank, non_blocking=True)

            img, seg = self.model(x, fg_mask)
            loss_dict = self.RGBLoss(img, gt[:, :3], False)
            if self.args.mode == 'xs2xs':
               loss_dict['ce_loss'] = self.args.ce_weight*self.SegLoss(seg, torch.argmax(gt[:,3:], dim=1))   
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
                if self.step % 100 == 0: 
                    image_set = self.prepare_image_set(data, img, seg)
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
                x, fg_mask, gt = self.get_input(data)
                size = x.size(0)
                x = x.cuda(self.args.rank, non_blocking=True)
                fg_mask = fg_mask.cuda(self.args.rank, non_blocking=True)
                gt = gt.cuda(self.args.rank, non_blocking=True)
                
                img, seg = self.model(x, fg_mask)

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

