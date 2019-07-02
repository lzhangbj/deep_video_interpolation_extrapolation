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
from losses import RGBLoss, PSNR, SSIM, IoU, GANLoss, GANMapLoss, VGGCosineLoss
import nets

from data import get_dataset
from utils.net_utils import *


def get_model(args):
    # build model
    model = nets.__dict__[args.model](args)
    return model


class GANer:
    def __init__(self, args):
        self.args = args

        args.logger.info('Initializing trainer')
        
        self.model = get_model(args)
        if args.load_G:
            for p in self.model.netG.parameters():
                p.requires_grad = False
        torch.cuda.set_device(args.rank)
        self.model.cuda(args.rank)
        self.model = torch.nn.parallel.DistributedDataParallel(self.model,
                device_ids=[args.rank])

        train_dataset, val_dataset = get_dataset(args)

        if not args.val:
            # train loss
            self.RGBLoss = RGBLoss(args).cuda(args.rank)
            self.SegLoss = nn.CrossEntropyLoss().cuda(args.rank)
            self.GANLoss = GANMapLoss().cuda(args.rank)
            # self.GANLoss = GANLoss(tensor=torch.FloatTensor).cuda(args.rank)
            self.GANFeatLoss = nn.L1Loss().cuda(args.rank)

            if args.optG == 'adamax':
                self.optG = torch.optim.Adamax(self.model.module.netG.parameters(), lr=args.lr_G)
            if args.optD == 'sgd':
                self.optD = torch.optim.SGD(self.model.module.netD.parameters(), lr=args.lr_D, momentum=0.9)
            elif args.optD == 'adamax':
                self.optD = torch.optim.Adamax(self.model.module.netD.parameters(), lr=args.lr_D)

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
        self.epoch = 1
        if args.load_G:
            self.load_temp()
        elif args.resume or ( args.val and not args.checkepoch_range):
            self.load_checkpoint()

        if args.rank == 0:
            if not args.load_G:
                if args.val:
                    self.writer =  SummaryWriter(args.path+'/val_logs') if args.interval == 2 else\
                                    SummaryWriter(args.path+'/val_int_1_logs')
                else:
                    self.writer = SummaryWriter(args.path+'/logs')
            else:
                if args.val:
                    self.writer =  SummaryWriter(args.path+'/dis_val_logs') if args.interval == 2 else\
                                    SummaryWriter(args.path+'/dis_val_int_1_logs')
                else:
                    self.writer = SummaryWriter(args.path+'/dis_{}_logs'.format(args.session))
        self.heatmap = self.create_stand_heatmap()

    def set_epoch(self, epoch):
        self.args.logger.info("Start of epoch %d" % (epoch+1))
        self.epoch = epoch + 1
        self.train_loader.sampler.set_epoch(epoch)
        # self.val_loader.sampler.set_epoch(epoch)

    def get_input(self, data):
        if self.args.mode == 'xs2xs':
            if self.args.syn_type == 'extra':
                x = torch.cat([data['frame1'], data['frame2'], data['seg1'], data['seg2']], dim=1)
                gt = torch.cat([data['frame3'], data['seg3']], dim=1)
            else:
                x = torch.cat([data['frame1'], data['frame3'], data['seg1'], data['seg3']], dim=1)
                gt = torch.cat([data['frame2'], data['seg2']], dim=1)      
        if self.args.mode == 'edge':
            if self.args.syn_type == 'extra':
                x = torch.cat([data['frame1'], data['frame2'], data['seg1'], data['seg2'], data['edge']], dim=1)
                gt = torch.cat([data['frame3'], data['seg3']], dim=1)
            else:
                x = torch.cat([data['frame1'], data['frame3'], data['seg1'], data['seg3'], data['edge']], dim=1)
                gt = torch.cat([data['frame2'], data['seg2']], dim=1)        
        elif self.args.mode == 'xss2x':
            if self.args.syn_type == 'extra':
                x = torch.cat([data['frame1'], data['frame2'], data['seg1'], data['seg2'], data['seg3']], dim=1)
                gt = data['frame3']   
            else:
                x = torch.cat([data['frame1'], data['frame3'], data['seg1'], data['seg2'], data['seg3']], dim=1)
                gt = data['frame2']   
        return x, gt   

    def normalize(self, img):
        return (img+1)/2

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


    def prepare_image_set(self, data, img, seg, pred_fake, pred_real, label_map, gen_prob=None, diff_map=None, extra=False):
        # print(pred_real[0][-1].size())
        view_rgbs = [   self.normalize(data['frame1'][0]), 
                        self.normalize(data['frame2'][0]), 
                        self.normalize(data['frame3'][0])   ]
        view_segs = [vis_seg_mask(data['seg'+str(i)][0].unsqueeze(0), 20).squeeze(0) for i in range(1, 4)]


        if not extra:
            pred_rgb = self.normalize(img[0].cpu())
            pred_seg = vis_seg_mask(seg[0].cpu().unsqueeze(0), 20).squeeze(0) if self.args.mode in ['xs2xs', 'edge'] else torch.zeros_like(view_segs[0])
            insert_index = 2 if self.args.syn_type == 'inter' else 3
            
            view_rgbs.insert(insert_index, pred_rgb)
            view_segs.insert(insert_index, pred_seg)

            view_probs_gt = []
            for i in range(self.args.num_D):
                toDraw = F.interpolate(pred_real[i][-1][0].unsqueeze(0).cpu(), (128, 256), mode='bilinear', align_corners=True).squeeze(0)
                view_probs_gt.append(self.create_heatmap(toDraw))
            view_probs_gt.append(self.heatmap)
            view_probs_gt.append(torch.zeros_like(self.heatmap))

            if diff_map is not None:
                view_probs_gt.append(self.create_heatmap(diff_map[0].cpu()))

            view_probs_fk = []
            for i in range(self.args.num_D):
                toDraw = F.interpolate(pred_fake[i][-1][0].unsqueeze(0).cpu(), (128, 256), mode='bilinear', align_corners=True).squeeze(0)
                view_probs_fk.append(self.create_heatmap(toDraw))
            view_probs_fk.append(self.create_heatmap(F.interpolate(label_map[0].unsqueeze(0).cpu(), (128, 256), mode='bilinear', align_corners=True).squeeze(0)))
            
            if gen_prob is not None:
                view_probs_fk.append(self.create_heatmap(gen_prob[0].cpu()))
            write_in_img = make_grid(view_rgbs + view_segs + view_probs_gt + view_probs_fk, nrow=4)
        else:
            pred_rgb = self.normalize(img[0][0].cpu())
            pred_seg = vis_seg_mask(seg[0][0].cpu().unsqueeze(0), 20).squeeze(0) if self.args.mode in ['xs2xs', 'edge'] else torch.zeros_like(view_segs[0])
            insert_index = 2 if self.args.syn_type == 'inter' else 3
            
            view_rgbs.insert(insert_index, pred_rgb)
            view_segs.insert(insert_index, pred_seg)

            view_probs_gt = []
            for i in range(self.args.num_D):
                toDraw = F.interpolate(pred_real[i][-1][0].unsqueeze(0).cpu(), (128, 256), mode='bilinear', align_corners=True).squeeze(0)
                view_probs_gt.append(self.create_heatmap(toDraw))
            view_probs_gt.append(self.heatmap)
            view_probs_gt.append(torch.zeros_like(self.heatmap))

            if diff_map is not None:
                view_probs_gt.append(self.create_heatmap(diff_map[0].cpu()))

            view_probs_fk = []
            for i in range(self.args.num_D):
                toDraw = F.interpolate(pred_fake[i][-1][0].unsqueeze(0).cpu(), (128, 256), mode='bilinear', align_corners=True).squeeze(0)
                view_probs_fk.append(self.create_heatmap(toDraw))
            view_probs_fk.append(self.create_heatmap(F.interpolate(label_map[0].unsqueeze(0).cpu(), (128, 256), mode='bilinear', align_corners=True).squeeze(0)))
            if gen_prob is not None:
                view_probs_fk.append(self.create_heatmap(gen_prob[0].cpu()))      

            # extra images
            view_pred_rgbs = []
            view_pred_segs = []
            for i in range(self.args.extra_length):
                pred_rgb = self.normalize(img[i][0].cpu())
                pred_seg = vis_seg_mask(seg[i].cpu(), 20).squeeze(0) if self.args.mode == 'xs2xs' else torch.zeros_like(view_segs[0])
                view_pred_rgbs.append(pred_rgb)
                view_pred_segs.append(pred_seg)      

            write_in_img = make_grid(view_rgbs + view_segs + view_probs_gt + view_probs_fk + view_pred_rgbs + view_pred_segs, nrow=4) 

        return write_in_img

    def GAN_feat_loss(self, pred_fake, pred_real):
        num_D = self.args.num_D
        n_layers_D = self.args.n_layer_D
        feat_weights = 1.0  / n_layers_D
        D_weights = 1.0 / num_D
        loss_G_GAN_Feat = 0
        for i in range(num_D):
            for j in range(len(pred_fake[i])-1):
                loss_G_GAN_Feat += D_weights * feat_weights * \
                    self.GANFeatLoss(pred_fake[i][j], pred_real[i][j].detach()) #* self.opt.feat_weight
        return self.args.adv_feat_weight*loss_G_GAN_Feat

    def create_diff_map(self, fake, gt):
        diff_map = torch.mean(torch.abs(fake.detach()-gt), dim=1, keepdim=True)

        diff_pooled = F.avg_pool2d(diff_map, kernel_size=23, stride=1, padding=11, count_include_pad=False)
        # print(diff_pooled.mean())
        diff_pooled/=0.1
        diff_pooled[diff_pooled > 1] = 1
        return diff_pooled

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
            x, gt = self.get_input(data)
            # x.requires_grad=True
            # gt.requires_grad=True
            x = x.cuda(self.args.rank, non_blocking=True)
            gt = gt.cuda(self.args.rank, non_blocking=True)

            img, seg, pred_fake_D, pred_real_D, pred_fake_G, label_map = self.model(x, gt)
            # diff_map = self.create_diff_map(img, gt[:,:3])
            if not self.args.load_G :#and not ( self.epoch == 1 and self.step % 50 != 0 ) :
                loss_dict = self.RGBLoss(img, gt[:, :3], False)
                if self.args.mode in ['xs2xs', 'edge']:
                   loss_dict['ce_loss'] = self.args.ce_weight*self.SegLoss(seg, torch.argmax(gt[:,3:], dim=1))

                # loss and accuracy
                loss = 0
                for i in loss_dict.values():
                    loss += i
                loss_dict['nonadv_loss'] = loss

                # loss_dict['prob_loss'] = 40*self.GANFeatLoss(prob, diff_map)

                # generator gan loss
                loss_dict['adv_loss'] = self.args.adv_weight*self.GANLoss(pred_fake_G, label_map, True, gen=True)
                # loss_dict['adv_feat_loss'] = 0 #self.args.adv_feat_weight * self.GAN_feat_loss(pred_fake_G, pred_real_D)
                # loss_dict['adv_loss'] = loss_dict['adv_loss'] + loss_dict['adv_feat_loss']

                loss_dict['g_loss'] = loss_dict['adv_loss'] + loss_dict['nonadv_loss'] #+ loss_dict['prob_loss']
            else:
                loss_dict = OrderedDict()

            # discriminator loss
            loss_dict['d_real_loss'] = self.args.d_weight*self.GANLoss(pred_real_D, label_map, True)
            loss_dict['d_fake_loss'] = self.args.d_weight*self.GANLoss(pred_fake_D, label_map, False)
            loss_dict['d_loss'] = loss_dict['d_real_loss'] + loss_dict['d_fake_loss']
            self.sync(loss_dict)


            self.optG.zero_grad()
            loss_dict['g_loss'].backward()
            if not self.args.load_G :#and not (self.step % 2 != 0 ):
                # generator backward pass
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
                        'load [{load_time:.3f}s] comp [{comp_time:.3f}s]'.format(
                            epoch=self.epoch, tot_epoch=self.args.epochs,
                            cur_batch=self.step+1, tot_batch=len(self.train_loader),
                            load_time=load_time, comp_time=comp_time
                        )
                    )
                    comp_time = 0
                    load_time = 0
                if self.step % 10 == 0: 
                    image_set = self.prepare_image_set(data, img, seg, pred_fake_D, pred_real_D, label_map) #, prob, diff_map)
                    self.writer.add_image('image_{}'.format(self.global_step), image_set, self.global_step)


    def get_dis_probs(self, tensor):
        bs = tensor[0][-1].size(0)
        probs = torch.zeros(bs, self.args.num_D).cuda(self.args.rank)
        for i in range(self.args.num_D):
            probs[:, i] = torch.mean(tensor[i][-1], dim=[1,2,3])
        return probs   

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
        for i in range(self.args.num_D):
            val_criteria['fake_mean_'+str(i)] = AverageMeter()
            val_criteria['real_mean_'+str(i)] = AverageMeter()
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

                img, seg, pred_fake, pred_real, t, label_map = self.model(x, gt)
                assert t is None

                fake_probs = self.get_dis_probs(pred_fake)
                real_probs = self.get_dis_probs(pred_real)

                # rgb criteria
                step_losses['l1'] =   self.L1Loss(img, gt[:,:3])
                step_losses['psnr'] = self.PSNRLoss((img+1)/2, (gt[:,:3]+1)/2)
                step_losses['ssim'] = 1-self.SSIMLoss(img, gt[:,:3])
                step_losses['iou'] =  self.IoULoss(torch.argmax(seg, dim=1), torch.argmax(gt[:,3:], dim=1))
                step_losses['vgg'] =  self.VGGCosLoss(img, gt[:, :3], False)

                for i in range(self.args.num_D):
                    step_losses['real_mean_'+str(i)] =  torch.mean(real_probs[:, i])
                    step_losses['fake_mean_'+str(i)] =  torch.mean(fake_probs[:, i])
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
                            img, seg, _, _, _, _ = self.model(x)
                        seg_fil = torch.argmax(seg, dim=1)
                        seg_fil = transform_seg_one_hot(seg_fil, 20, cuda=True)*2-1
                        imgs.append(img)
                        segs.append(seg_fil)

                # save validate result
                # p = torch.cat([frame1.cuda(), frame_middle, frame2.cuda(), img], dim=1)
                # p = p.cpu().detach().numpy()
                # np.save('../predict/val_'+str(end)+'_'+str(i).zfill(6)+'.npy', p)

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
                            image_set = self.prepare_image_set(data, img, seg, pred_fake, pred_real, label_map)
                        else:
                            image_set = self.prepare_image_set(data, imgs, segs, pred_fake, pred_real, label_map, extra=True)
                        image_name = 'e{}_img_{}'.format(self.epoch, self.step)
                        # for i in range(self.args.num_D):
                        #     image_name += '_f{:.3f}_r{:.3f}'.format(fake_probs[0,i].item(), real_probs[0,i].cpu().item())
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
            for i in range(self.args.num_D):
                self.args.logger.info('{}\t: {:.4f}\n \
                {}\t: {:.4f}\n'.format('fake_mean_'+str(i), val_criteria['fake_mean_'+str(i)].avg, 
                       'real_mean_'+str(i), val_criteria['real_mean_'+str(i)].avg))
            tfb_info = {key:value.avg for key,value in val_criteria.items()}
            self.writer.add_scalars('val/score', tfb_info, self.epoch)

    def sync(self, loss_dict, mean=True):
        '''Synchronize all tensors given using mean or sum.'''
        for tensor in loss_dict.values():
            dist.all_reduce(tensor)
            if mean:
                tensor.div_(self.args.gpus)

    def save_checkpoint(self):
        if not self.args.load_G:
            save_md_dir = '{}_{}_{}_{}_{}'.format(self.args.netG, self.args.netD, self.args.mode, self.args.syn_type, self.args.session)
        else:
            save_md_dir = '{}_{}_dis_{}_{}_{}'.format(self.args.netG, self.args.netD, self.args.mode, self.args.syn_type, self.args.session)

        save_name = os.path.join(self.args.path, 
                                'checkpoint',
                                save_md_dir + '_{}_{}.pth'.format(self.epoch, self.step))
        self.args.logger.info('Saving checkpoint..')
        torch.save({
            'session': self.args.session,
            'epoch': self.epoch + 1,
            'netG': self.model.module.netG.state_dict(),
            'netD': self.model.module.netD.state_dict(),
            'optG': self.optG.state_dict(),
            'optD': self.optD.state_dict(),
        }, save_name)
        self.args.logger.info('save model: {}'.format(save_name))
     
    def load_temp(self):
        if not self.args.load_GANG:
            load_md_dir = '{}_{}_{}_{}'.format(self.args.netG, self.args.mode, self.args.syn_type, self.args.checksession)
            if self.args.load_dir is not None:
                load_name = os.path.join(self.args.load_dir,
                                        'checkpoint',
                                        load_md_dir+'_{}_{}.pth'.format(self.args.checkepoch, self.args.checkpoint))
            else:
                load_name = os.path.join(load_md_dir+'_{}_{}.pth'.format(self.args.checkepoch, self.args.checkpoint))
            self.args.logger.info('Loading checkpoint %s' % load_name)
            ckpt = torch.load(load_name)
            self.model.module.netG.load_state_dict(ckpt['model'])
        else:
            load_md_dir = '{}_{}_{}_{}_{}'.format(self.args.netG, "multi_scale_img", self.args.mode, self.args.syn_type, self.args.checksession)
            if self.args.load_dir is not None:
                load_name = os.path.join(self.args.load_dir,
                                        'checkpoint',
                                        load_md_dir+'_{}_{}.pth'.format(self.args.checkepoch, self.args.checkpoint))
            else:
                load_name = os.path.join(load_md_dir+'_{}_{}.pth'.format(self.args.checkepoch, self.args.checkpoint))
            self.args.logger.info('Loading checkpoint %s' % load_name)
            ckpt = torch.load(load_name)
            self.model.module.netG.load_state_dict(ckpt['netG'])          
        # transfer opt params to current device
        for state in self.optG.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda(self.args.rank)

        self.args.logger.info('checkpoint loaded')    

    def load_checkpoint(self):
        load_md_dir = '{}_{}_{}_{}_{}'.format(self.args.netG, self.args.netD, self.args.mode, self.args.syn_type, self.args.checksession)
        if self.args.load_dir is not None:
            load_name = os.path.join(self.args.load_dir,
                                    'checkpoint',
                                    load_md_dir+'_{}_{}.pth'.format(self.args.checkepoch, self.args.checkpoint))
        else:
            load_name = os.path.join(load_md_dir+'_{}_{}.pth'.format(self.args.checkepoch, self.args.checkpoint))
        self.args.logger.info('Loading checkpoint %s' % load_name)
        ckpt = torch.load(load_name)
        self.model.module.netG.load_state_dict(ckpt['netG'])
        self.model.module.netD.load_state_dict(ckpt['netD'])
        if not self.args.val:
            self.epoch = ckpt['epoch']
            self.global_step = (self.epoch-1)*len(self.train_loader)
            self.optG.load_state_dict(ckpt['optG'])
            self.optD.load_state_dict(ckpt['optD'])
            # transfer opt params to current device
            for state in self.optG.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda(self.args.rank)
            for state in self.optD.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda(self.args.rank)
        else:
            assert ckpt['epoch']-1 == self.args.checkepoch, [ckpt['epoch'], self.args.checkepoch]
            self.epoch = ckpt['epoch'] - 1

        self.args.logger.info('checkpoint loaded')




# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3,4,5 python main.py  --disp_interval 100 --mode xs2xs --syn_type inter --bs 48 --nw 8 --ce_w 30  --s 7  gan --adv_w 20 --d_w 10 --netD motion_img --lrD 0.001 --n_layer_D 1 --numD 1 --oD adamax