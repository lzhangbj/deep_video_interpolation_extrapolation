import os
import sys
from time import time
import math
import argparse
import itertools
import shutil
from collections import OrderedDict

import glob
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
from utils.net_utils import *
# from cfg import cfg

def get_model(args):
    # build model
    model = nets.__dict__[args.model](args)
    return model


class RefinerGAN:
    def __init__(self, args):
        self.args = args

        args.logger.info('Initializing trainer')
        # if not os.path.isdir('../predict'):       only used in validation
        #     os.makedirs('../predict')
        self.model = get_model(args)
        if self.args.lock_coarse:
            for p in self.model.coarse_model.parameters():
                p.requires_grad = False
        torch.cuda.set_device(args.rank)
        self.model.cuda(args.rank)
        self.model = torch.nn.parallel.DistributedDataParallel(self.model,
                device_ids=[args.rank])
        train_dataset, val_dataset = get_dataset(args)

        if not args.val:
            # train loss
            self.coarse_RGBLoss = RGBLoss(args, sharp=False)
            self.refine_RGBLoss = RGBLoss(args, sharp=False, refine=True)
            self.SegLoss = nn.CrossEntropyLoss()
            self.GANLoss = GANLoss(tensor=torch.FloatTensor)

            self.coarse_RGBLoss.cuda(args.rank)
            self.refine_RGBLoss.cuda(args.rank)
            self.SegLoss.cuda(args.rank)
            self.GANLoss.cuda(args.rank)

            if args.optimizer == "adamax":
                self.optG = torch.optim.Adamax(list(self.model.module.coarse_model.parameters()) + list(self.model.module.refine_model.parameters()), lr=args.learning_rate)
            elif args.optimizer == "adam":
                self.optG = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)
            elif args.optimizer == "sgd":
                self.optG = torch.optim.SGD(self.model.parameters(), lr=args.learning_rate, momentum=0.9)

            # self.optD = torch.optim.Adam(self.model.module.discriminator.parameters(), lr=args.learning_rate)
            self.optD = torch.optim.SGD(self.model.module.discriminator.parameters(), lr=args.learning_rate, momentum=0.9)


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
        self.heatmap = self.create_stand_heatmap()

    def prepare_heat_map(self, prob_map):
        bs, c, h, w = prob_map.size()
        if h!=128:
            prob_map_ = F.interpolate(prob_map, size=(128, 256), mode='nearest', align_corners=True)
        return prob_map

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

    def create_stand_heatmap(self):
        heatmap = torch.zeros(3, 128, 256)
        for i in range(256):
            heatmap[0, :, i] = max(0, 1 - 2.*i/256)
            heatmap[1, :, i] = max(0, 2.*i/256 - 1)
            heatmap[2, :, i] = 1-heatmap[0, :, i]-heatmap[1, :, i]
        return heatmap


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

    def prepare_image_set(self, data, coarse_img, refined_imgs, seg, pred_fake, pred_real, extra=False):
        view_rgbs = [   self.normalize(data['frame1'][0]), 
                        self.normalize(data['frame2'][0]), 
                        self.normalize(data['frame3'][0])   ]
        view_segs = [vis_seg_mask(data['seg'+str(i)][0].unsqueeze(0), 20).squeeze(0) for i in range(1, 4)]


        # gan
        view_probs = []
        view_probs.append(self.heatmap)

        for i in range(self.args.num_D):
            toDraw = F.interpolate(pred_real[i][-1][0].unsqueeze(0).cpu(), (128, 256), mode='bilinear', align_corners=True).squeeze(0)
            view_probs.append(self.create_heatmap(toDraw))
            toDraw = F.interpolate(pred_fake[i][-1][0].unsqueeze(0).cpu(), (128, 256), mode='bilinear', align_corners=True).squeeze(0)
            view_probs.append(self.create_heatmap(toDraw))        

        if not extra:
            # coarse
            pred_rgb = self.normalize(coarse_img[0])
            pred_seg = vis_seg_mask(seg[0].unsqueeze(0), 20).squeeze(0) if self.args.mode == 'xs2xs' else torch.zeros_like(view_segs[0])
            insert_index = 2 if self.args.syn_type == 'inter' else 3
            
            view_rgbs.insert(insert_index, pred_rgb)
            view_segs.insert(insert_index, pred_seg)
            view_segs.append(torch.zeros_like(view_segs[-1]))
            # refine
            refined_bs_imgs = [ refined_img[0].unsqueeze(0) for refined_img in refined_imgs ] 
            for i in range(self.args.n_scales):
                insert_img = F.interpolate(refined_bs_imgs[i], size=(128,256))[0].clamp_(-1, 1) 

                pred_rgb = self.normalize(insert_img)
                insert_ind = insert_index + i+1
                view_rgbs.insert(insert_ind, pred_rgb)

            write_in_img = make_grid(view_rgbs + view_segs + view_probs, nrow=4+self.args.n_scales)
        # else:
        #     view_rgbs.insert(3, torch.zeros_like(view_rgbs[-1]))
        #     view_segs.insert(3, torch.zeros_like(view_segs[-1]))

        #     view_pred_rgbs = []
        #     view_pred_segs = []
        #     for i in range(self.args.extra_length):
        #         pred_rgb = self.normalize(img[i][0].cpu())
        #         pred_seg = vis_seg_mask(seg[i].cpu(), 20).squeeze(0) if self.args.mode == 'xs2xs' else torch.zeros_like(view_segs[0])
        #         view_pred_rgbs.append(pred_rgb)
        #         view_pred_segs.append(pred_seg)
        #     write_in_img = make_grid(view_rgbs + view_segs + view_pred_rgbs + view_pred_segs, nrow=4)

        
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

            coarse_img, refined_imgs, seg, pred_fake_D, pred_real_D, pred_fake_G = self.model(x, fg_mask, gt)
            if not self.args.lock_coarse:
                loss_dict = self.coarse_RGBLoss(coarse_img, gt[:, :3], False)
                if self.args.mode == 'xs2xs':
                   loss_dict['ce_loss'] = self.args.ce_weight*self.SegLoss(seg, torch.argmax(gt[:,3:], dim=1))   
            else:
                loss_dict = OrderedDict()
            for i in range(self.args.n_scales):
                # print(i, refined_imgs[-i].size())
                loss_dict.update(self.refine_RGBLoss(refined_imgs[-i-1], F.interpolate(gt[:,:3], scale_factor=(1/2)**i, mode='bilinear', align_corners=True),\
                                                                     refine_scale=1/(2**i), step=self.global_step, normed=False))
            # loss and accuracy
            loss = 0
            for i in loss_dict.values():
                loss += torch.mean(i)
            loss_dict['loss_all'] = loss

            if self.global_step > 1000:
                loss_dict['adv_loss'] = self.args.refine_adv_weight*self.GANLoss(pred_fake_G, True)
                
                g_loss = loss_dict['loss_all'] + loss_dict['adv_loss']

                loss_dict['d_real_loss'] = self.args.refine_d_weight*self.GANLoss(pred_real_D, True)
                loss_dict['d_fake_loss'] = self.args.refine_d_weight*self.GANLoss(pred_fake_D, False)
                loss_dict['d_loss'] = loss_dict['d_real_loss'] + loss_dict['d_fake_loss']

            else:
                g_loss = loss_dict['loss_all'] 

                loss_dict['d_real_loss'] = 0*self.GANLoss(pred_real_D, True)
                loss_dict['d_fake_loss'] = 0*self.GANLoss(pred_fake_D, False)
                loss_dict['d_loss'] = loss_dict['d_real_loss'] + loss_dict['d_fake_loss']               

            self.sync(loss_dict)

            self.optG.zero_grad()
            g_loss.backward()
            self.optG.step()

            # discriminator backward pass
            self.optD.zero_grad()
            loss_dict['d_loss'].backward()
            self.optD.step()
            comp_time += time() - end
            end = time()

            if self.args.rank == 0:
                # add info to tensorboard
                info = {key:value.item() for key,value in loss_dict.items()}
                # add discriminator value
                pred_value = 0
                real_value = 0
                for i in range(self.args.num_D):
                    pred_value += torch.mean(pred_fake_D[i][-1])
                    real_value += torch.mean(pred_real_D[i][-1])
                pred_value/=self.args.num_D
                real_value/=self.args.num_D
                info["fake"] = pred_value.item()
                info["real"] = real_value.item()
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
                    image_set = self.prepare_image_set(data, coarse_img.cpu(), [ refined_img.cpu() for refined_img in refined_imgs], seg.cpu(), \
                                            pred_fake_D, pred_real_D)
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
                
                coarse_img, refined_imgs, seg, pred_fake_D, pred_real_D= self.model(x, fg_mask, gt)
                # rgb criteria
                step_losses['l1'] =   self.L1Loss(refined_imgs[-1], gt[:,:3])
                step_losses['psnr'] = self.PSNRLoss((refined_imgs[-1]+1)/2, (gt[:,:3]+1)/2)
                step_losses['ssim'] = 1-self.SSIMLoss(refined_imgs[-1], gt[:,:3])
                step_losses['iou'] =  self.IoULoss(torch.argmax(seg, dim=1), torch.argmax(gt[:,3:], dim=1))
                step_losses['vgg'] =  self.VGGCosLoss(refined_imgs[-1], gt[:, :3], False)
                self.sync(step_losses) # sum
                for key in list(val_criteria.keys()):
                    val_criteria[key].update(step_losses[key].cpu().item(), size*self.args.gpus)

                if self.args.syn_type == 'extra': # not implemented
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
                            image_set = self.prepare_image_set(data, coarse_img.cpu(), [ refined_img.cpu() for refined_img in refined_imgs], seg.cpu(), \
                                            pred_fake_D, pred_real_D)
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

    def test(self):
        self.args.logger.info('testing started')
        self.model.eval()

        with torch.no_grad():
            end = time()
            load_time = 0
            comp_time = 0

            img_count = 0
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

                bs = img.size(0)
                for i in range(bs):
                    pred_img = self.normalize(img[i])
                    gt_img = self.normalize(gt[i, :3])

                    save_img(pred_img, '{}/{}_pred.png'.format(self.args.save_dir, img_count))
                    save_img(gt_img, '{}/{}_gt.png'.format(self.args.save_dir, img_count))
                    img_count+=1

                comp_time += time() - end
                end = time()

                # print
                if self.args.rank == 0:
                    if self.step % self.args.disp_interval == 0:
                        self.args.logger.info(
                            'img [{}] load [{load_time:.3f}s] comp [{comp_time:.3f}s]'.format(img_count,
                                load_time=load_time, comp_time=comp_time
                            )
                        )
                        comp_time = 0
                        load_time = 0


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
            'optG': self.optG.state_dict(),
            'optD': self.optD.state_dict()
        }, save_name)
        self.args.logger.info('save model: {}'.format(save_name))

    def load_checkpoint(self):
        load_md_dir = '{}_{}_{}_{}'.format("RefineNet", self.args.mode, self.args.syn_type, self.args.checksession)
        if self.args.load_dir is not None:
            load_name = os.path.join(self.args.load_dir,
                                    'checkpoint',
                                    load_md_dir+'_{}_{}.pth'.format(self.args.checkepoch, self.args.checkpoint))
        else:
            load_name = os.path.join(load_md_dir+'_{}_{}.pth'.format(self.args.checkepoch, self.args.checkpoint))
        self.args.logger.info('Loading checkpoint %s' % load_name)
        ckpt = torch.load(load_name)
        if self.args.lock_coarse:
            model_dict = self.model.module.state_dict()
            new_ckpt = OrderedDict()
            for key,item in ckpt['model'].items():
                if 'coarse' in key:
                    new_ckpt[key] = item
            model_dict.update(new_ckpt)
            self.model.module.load_state_dict(model_dict)
        else:
            self.model.module.load_state_dict(ckpt['model'])
        # transfer opt params to current device
        if not self.args.lock_coarse:
            if not self.args.val :
                self.optimizer.load_state_dict(ckpt['optimizer'])
                self.epoch = ckpt['epoch']
                self.global_step = (self.epoch-1)*len(self.train_loader)
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda(self.args.rank)
            else :
                assert ckpt['epoch']-1 == self.args.checkepoch, [ckpt['epoch'], self.args.checkepoch]
                self.epoch = ckpt['epoch'] - 1
        self.args.logger.info('checkpoint loaded')

