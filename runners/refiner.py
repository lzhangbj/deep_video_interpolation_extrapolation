import os
import sys
from time import time
import math
import argparse
from PIL import Image
import itertools
import shutil
from collections import OrderedDict
import glob
import pickle
import numpy as np
from torch.autograd import Variable
import torch
import cv2
import torch.nn as nn
import torch.distributed as dist 
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets 
from tensorboardX import SummaryWriter 
from torchvision.utils import make_grid, save_image

from losses import RGBLoss, PSNR, SSIM, IoU, GANLoss, VGGCosineLoss
import nets
# from flow_process import imgs2vid

from data import get_dataset
from folder import rgb_load, seg_load
from utils.net_utils import *

def get_model(args):
	# build model
	model = nets.__dict__[args.model](args)
	return model



def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Refiner:
	def __init__(self, args):
		self.args = args

		args.logger.info('Initializing trainer')
		# if not os.path.isdir('../predict'):       only used in validation
		#     os.makedirs('../predict')
		self.model = get_model(self.args)
		if self.args.lock_coarse or self.args.lock_retrain or self.args.lock_low:
			for p in self.model.coarse_model.parameters():
				p.requires_grad = False
		if self.args.lock_low or self.args.lock_refine:
			for p in self.model.refine_model.parameters():
				p.requires_grad = False
		coarse_params_cnt = count_parameters(self.model.coarse_model)
		print("coarse params ", coarse_params_cnt)
		refine_params_cnt = count_parameters(self.model.refine_model)
		print("refine params ", refine_params_cnt)
		if self.args.re_ref:
			re_ref_params_cnt = count_parameters(self.model.re_ref_model)
			print("re_ref params ", re_ref_params_cnt)
		torch.cuda.set_device(args.rank)
		self.model.cuda(args.rank)
		self.model = torch.nn.parallel.DistributedDataParallel(self.model,
				device_ids=[args.rank])
		if self.args.split not in ['cycgen', 'mycycgen']:
			train_dataset, val_dataset = get_dataset(args)

		if args.split == 'train':
			# train loss
			self.coarse_RGBLoss = RGBLoss(args, sharp=False)
			self.refine_RGBLoss = RGBLoss(args, sharp=False, refine=True)
			self.SegLoss = nn.CrossEntropyLoss()
			self.coarse_RGBLoss.cuda(args.rank)
			self.refine_RGBLoss.cuda(args.rank)
			self.SegLoss.cuda(args.rank)

			if args.optimizer == "adamax":
				self.coarse_optimizer = torch.optim.Adamax(list(self.model.module.coarse_model.parameters()), lr=args.learning_rate)
				self.refine_optimizer = torch.optim.Adamax(list(self.model.module.refine_model.parameters()), lr=args.learning_rate)
			elif args.optimizer == "adam":
				self.coarse_optimizer = torch.optim.Adam(self.model.module.coarse_model.parameters(), lr=args.learning_rate)
				self.refine_optimizer = torch.optim.Adam(self.model.module.coarse_model.parameters(), lr=args.learning_rate)
			elif args.optimizer == "sgd":
				# to do
				self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.learning_rate, momentum=0.9)

			if self.args.high_res:
				self.high_res_optimizer = torch.optim.Adamax(list(self.model.module.high_res_model.parameters()), lr=args.learning_rate)

			if self.args.re_ref:
				self.re_ref_optimizer = torch.optim.Adamax(list(self.model.module.re_ref_model.parameters()), lr=args.learning_rate)



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
		if args.resume or (args.split != 'train' and not args.checkepoch_range) or args.pretrained_coarse or args.pretrained_low:
			self.load_checkpoint()

		if args.rank == 0:
			writer_name = args.path+'/{}_int_{}_len_{}_{}_logs'.format(self.args.split, self.args.interval, self.args.vid_length, self.args.dataset)
			self.writer = SummaryWriter(writer_name)

		if self.args.re_ref:
			self.base_flow_map = self.standard_flow_map().unsqueeze(0)
			self.base_flow_map = F.interpolate(self.base_flow_map, size=(128, 256), mode='bilinear', align_corners=True)[0]

		self.stand_heat_map = self.create_stand_heatmap()

	def set_epoch(self, epoch):
		self.args.logger.info("Start of epoch %d" % (epoch+1))
		self.epoch = epoch + 1
		self.train_loader.sampler.set_epoch(epoch)
		# self.val_loader.sampler.set_epoch(epoch)

	def get_input(self, data):
		if self.args.syn_type == 'extra':
			if self.args.fix_init_frames:
				x = torch.cat([data['frame1'], data['frame2']], dim=1)
				seg = torch.cat([data['seg1'], data['seg2']], dim=1) if self.args.mode == 'xs2xs' else None
				mask = torch.cat([data['fg_mask1'],data['fg_mask2']], dim=1) if self.args.mode == 'xs2xs' else None
				gt_x = [ data['frame'+str(i+3)] for i in range(self.args.pred_len) ]
				gt_seg = [ data['seg'+str(i+3)] if self.args.mode == 'xs2xs' else None for i in range(self.args.pred_len) ]
			else:
				x = torch.cat([data['frame1'], data['frame2']], dim=1)
				seg = torch.cat([data['seg1'], data['seg2']], dim=1) if self.args.mode == 'xs2xs' else None
				mask = torch.cat([data['fg_mask1'],data['fg_mask2']], dim=1) if self.args.mode == 'xs2xs' else None
				gt_x = data['frame3']
				gt_seg = data['seg3'] if self.args.mode == 'xs2xs' else None
		else:
			x = torch.cat([data['frame1'], data['frame3']], dim=1)
			seg = torch.cat([data['seg1'], data['seg3']], dim=1) if self.args.mode == 'xs2xs' else None
			mask = torch.cat([data['fg_mask1'],data['fg_mask3']], dim=1) if self.args.mode == 'xs2xs' else None
			gt_x = data['frame2'] 
			gt_seg = data['seg2']  if self.args.mode == 'xs2xs' else None
		return x, seg, mask, gt_x, gt_seg  

	def get_inputs(self, data):
		assert self.args.syn_type == 'extra'
		x = torch.cat([data['frame1'], data['frame2']], dim=1)
		seg = torch.cat([data['seg1'], data['seg2']], dim=1) if self.args.mode == 'xs2xs' else None
		mask = torch.cat([data['fg_mask1'],data['fg_mask2']], dim=1) if self.args.mode == 'xs2xs' else None
		gt_x = [ data['frame'+str(i+3)] for i in range(self.args.vid_length) ]
		gt_seg = [ data['seg'+str(i+3)] if self.args.mode == 'xs2xs' else None for i in range(self.args.vid_length) ]
		return x, seg, mask, gt_x, gt_seg

	def normalize(self, img):
		return (img+1)/2

	def standard_flow_map(self):
		W = self.model.module.re_ref_model.W
		H = self.model.module.re_ref_model.H
		w = self.model.module.re_ref_model.w
		h = self.model.module.re_ref_model.h

		offset = torch.zeros(1,2,H,W)
		h_unit = H//h # 3
		w_unit = W//w # 3
		for i in range(h):
			offset[:,1,i*h_unit:min((i+1)*h_unit, H), :] = i
		for i in range(w):
			offset[:,0,:, i*w_unit:min((i+1)*w_unit, W)] = i

		h_w_add = torch.zeros(1, 2, H, W)
		h_w_add[:, 0] = w//2
		h_w_add[:, 1] = h//2
		offset = offset - h_w_add

		flow_map = self.flow_to_image(flow=offset[0])
		return flow_map

	def flow_to_image(self, flow):
		c, h, w = flow.size()
		flow_np = flow.permute(1,2,0).contiguous().numpy()
		hsv = np.zeros((h, w, 3), dtype=np.uint8)
		hsv[..., 1] = 255

		mag, ang = cv2.cartToPolar(flow_np[..., 0], flow_np[..., 1])
		hsv[..., 0] = ang * 180 / np.pi / 2
		hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
		bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
		bgr = bgr.astype(np.float32)/255
		return torch.tensor(bgr).permute(2,0,1).contiguous()

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

	def prepare_extra_image_set(self, data, imgs, segs, masks=None, inpaints=None):
		assert len(imgs) == self.args.num_pred_step*self.args.num_pred_once
		num_pred_imgs = self.args.num_pred_step*self.args.num_pred_once
		view_gt_rgbs = [ self.normalize(data['frame'+str(i+1)][0]) for i in range(num_pred_imgs+2)]
		view_gt_segs = [ vis_seg_mask(data['seg'+str(i+1)][0].unsqueeze(0), 20).squeeze(0) for i in range(num_pred_imgs+2)]

		black_img = torch.zeros_like(view_gt_rgbs[0])

		view_pred_imgs =  [black_img]*2
		view_pred_imgs += [self.normalize(img[0]) for img in imgs]

		view_pred_segs =  [black_img]*2
		view_pred_segs += [vis_seg_mask(seg[0].unsqueeze(0), 20).squeeze(0) for seg in segs]

		view_imgs = view_gt_rgbs + view_pred_imgs + view_gt_segs + view_pred_segs

		if self.args.inpaint:
			view_inpaint_imgs =  [black_img]*2
			view_inpaint_imgs += [self.normalize(img[0]) for img in inpaints]

			view_inpaint_masks =  [black_img, self.stand_heat_map]
			view_inpaint_masks += [ self.create_heatmap(img[0]) for img in masks]

			view_imgs+=view_inpaint_imgs
			view_imgs+=view_inpaint_masks

		write_in_img = make_grid(view_imgs, nrow=num_pred_imgs+2)

		return write_in_img

	def prepare_image_set(self, data, coarse_img, refined_imgs, seg, high_res_img=None, re_ref_img=None, flow=None):
		if self.args.dataset != 'vimeo':
			re_size = (128, 256)
		else:
			re_size = (256, 448)
		view_rgbs = [   self.normalize(data['frame1'][0]), 
						self.normalize(data['frame2'][0]), 
						self.normalize(data['frame3'][0])   ]
		if self.args.mode=='xs2xs':
			view_segs = [vis_seg_mask(data['seg'+str(i)][0].unsqueeze(0), 20).squeeze(0) for i in range(1, 4)]

		black_img = torch.zeros_like(view_rgbs[0])

		# if not extra:
		insert_index = 2 #if self.args.syn_type == 'inter' else 3

		# coarse
		pred_rgb = self.normalize(coarse_img[0])
		view_rgbs.insert(insert_index, pred_rgb)

		if self.args.mode=='xs2xs':
			pred_seg = vis_seg_mask(seg[0].unsqueeze(0), 20).squeeze(0) if self.args.mode == 'xs2xs' else torch.zeros_like(view_segs[0])
			view_segs.insert(insert_index, pred_seg)
		insert_ind = insert_index + 1

		view_imgs = view_rgbs 
		if self.args.mode=='xs2xs':
			view_imgs = view_imgs + view_segs

		re_size = re_size
		view_imgs = [F.interpolate(img.unsqueeze(0), size=re_size, mode='bilinear', align_corners=True)[0] for img in view_imgs]
		n_rows = 4+1

		if flow is not None:
			flow_f = self.flow_to_image(flow[0,0].cpu()).unsqueeze(0)
			flow_f = F.interpolate(flow_f, size=re_size, mode='bilinear', align_corners=True)[0]
			flow_b = self.flow_to_image(flow[0,1].cpu()).unsqueeze(0)
			flow_b = F.interpolate(flow_b, size=re_size, mode='bilinear', align_corners=True)[0]
			view_imgs += [flow_f,  flow_b, self.base_flow_map]

		write_in_img = make_grid(view_imgs, nrow=n_rows)

		return write_in_img

	def prepare_image_sets(self, data, coarse_imgs, refined_imgs, segs, high_res_imgs=None, re_ref_imgs=None):
		nrows = self.args.vid_length+2
		if self.args.dataset != 'vimeo':
			re_size = (128, 256)
		else:
			re_size = (256, 448)
		view_gt_rgbs = []
		
		for i in range(nrows):
			view_gt_rgbs.append(self.normalize(data['frame{:d}'.format(i+1)][0]))

		black_img = torch.zeros_like(view_gt_rgbs[0])

		# coarse
		coarse_rgbs = [black_img]*2 + [ self.normalize(coarse_imgs[i][0]) for i in range(nrows-2)]
		refine_rgbs = [black_img]*2 + [ self.normalize(refined_imgs[i][0]) for i in range(nrows-2)]
		view_imgs = view_gt_rgbs + refine_rgbs + coarse_rgbs
		if self.args.re_ref:
			re_ref_imgs = [black_img]*2 + [ self.normalize(re_ref_imgs[i][0]) for i in range(nrows-2)]
			view_imgs = view_gt_rgbs + re_ref_imgs + refine_rgbs + coarse_rgbs

		if self.args.mode=='xs2xs':
			view_gt_segs = [vis_seg_mask(data['seg'+str(i+1)][0].unsqueeze(0), 20).squeeze(0) for i in range(nrows)]
			pred_segs = [black_img]*2 + [ vis_seg_mask(segs[i][0].unsqueeze(0), 20).squeeze(0) for i in range(nrows-2)]
			view_segs = view_gt_segs + pred_segs
			view_imgs = view_segs + view_imgs

		write_in_img = make_grid(view_imgs, nrow=nrows)
		
		return write_in_img

	def get_loss_record_dict(self,prefix=''):
		D = {'{}_data_cnt'.format(prefix):0,
			'{}_all_loss_record'.format(prefix):0}
		if self.args.syn_type == 'extra':
			for i in range(self.args.num_pred_step*self.args.num_pred_once):
				d = {
				'{}_frame_{}_coarse_l1_loss_record'.format(prefix, i+1):0,
				'{}_frame_{}_coarse_ssim_loss_record'.format(prefix, i+1):0,
				'{}_frame_{}_coarse_gdl_loss_record'.format(prefix, i+1):0,
				'{}_frame_{}_coarse_vgg_loss_record'.format(prefix, i+1):0,
				'{}_frame_{}_coarse_ce_loss_record'.format(prefix, i+1):0,
				'{}_frame_{}_coarse_all_loss_record'.format(prefix, i+1):0
				}
				D.update(d)
				if self.args.inpaint:
					d = {
					'{}_frame_{}_inpaint_l1_loss_record'.format(prefix, i+1):0,
					'{}_frame_{}_inpaint_gdl_loss_record'.format(prefix, i+1):0,
					'{}_frame_{}_inpaint_mask_loss_record'.format(prefix, i+1):0,
					'{}_frame_{}_inpaint_all_loss_record'.format(prefix, i+1):0
					}
					D.update(d)
		else:
			d = {
				'{}_coarse_l1_loss_record'.format(prefix):0,
				'{}_coarse_ssim_loss_record'.format(prefix):0,
				'{}_coarse_gdl_loss_record'.format(prefix):0,
				'{}_coarse_vgg_loss_record'.format(prefix):0,
				'{}_coarse_ce_loss_record'.format(prefix):0,
				'{}_coarse_all_loss_record'.format(prefix):0
				}
			D.update(d)
		return D

	def update_loss_record_dict(self, record_dict, loss_dict, batch_size):
		record_dict['step_data_cnt']+=batch_size
		if self.args.syn_type == 'extra':
			for i in range(self.args.num_pred_step):
				for j in range(self.args.num_pred_once):
					frame_ind = 1+i*self.args.num_pred_once + j
					for loss_name in ['l1', 'gdl', 'ssim', 'vgg', 'ce']:
						record_dict['step_frame_{}_coarse_{}_loss_record'.format(frame_ind, loss_name)] += \
										batch_size*loss_dict['step_{}_frame_{}_{}_loss'.format(i, j, loss_name)].item()
						record_dict['step_frame_{}_coarse_all_loss_record'.format(frame_ind)] += \
										batch_size*loss_dict['step_{}_frame_{}_{}_loss'.format(i, j, loss_name)].item()
					if self.args.inpaint:
						frame_ind = 1+i*self.args.num_pred_once + j
						for loss_name in ['l1', 'gdl', 'mask']:
							record_dict['step_frame_{}_inpaint_{}_loss_record'.format(frame_ind, loss_name)] += \
											batch_size*loss_dict['step_{}_frame_{}_inpaint_{}_loss'.format(i, j, loss_name)].item()
							record_dict['step_frame_{}_inpaint_all_loss_record'.format(frame_ind)] += \
											batch_size*loss_dict['step_{}_frame_{}_inpaint_{}_loss'.format(i, j, loss_name)].item()

		else:
			for loss_name in ['l1', 'gdl', 'ssim', 'vgg', 'ce']:
				record_dict['step_coarse_{}_loss_record'.format(loss_name)] += \
											batch_size*loss_dict['step_{}_frame_{}_{}_loss'.format(i, j, loss_name)].item()
				record_dict['step_coarse_all_loss_record'.format(frame_ind)] += \
										batch_size*loss_dict['step_{}_frame_{}_{}_loss'.format(i, j, loss_name)].item()
		record_dict['step_all_loss_record']+=batch_size*loss_dict['loss_all'].item()
		return record_dict



	def train(self):
		self.args.logger.info('Training started')
		self.model.train()
		end = time()
		load_time = 0
		comp_time = 0

		step_loss_record_dict = self.get_loss_record_dict('step')
		epoch_loss_record_dict = self.get_loss_record_dict('epoch')

		for step, data in enumerate(self.train_loader):
			self.step = step
			load_time += time() - end
			end = time()
			# for tensorboard
			self.global_step += 1

			batch_size_p = data['frame1'].size(0)
			if self.args.syn_type == 'inter':
				x, seg, fg_mask, gt_x, gt_seg = self.get_input(data)
				x = x.cuda(self.args.rank, non_blocking=True)
				seg = seg.cuda(self.args.rank, non_blocking=True)  if self.args.mode == 'xs2xs' else None
				fg_mask = fg_mask.cuda(self.args.rank, non_blocking=True) if self.args.mode == 'xs2xs' else None
				gt_seg = gt_seg.cuda(self.args.rank, non_blocking=True) if self.args.mode == 'xs2xs' else None
				gt_x = gt_x.cuda(self.args.rank, non_blocking=True)
				

				coarse_img, refined_imgs, seg, attn_flow = self.model(x, seg=seg, gt_x=gt_x, gt_seg=gt_seg)
				scaled_gt_x = gt_x
				if not self.args.lock_coarse and not self.args.lock_retrain:
					loss_dict = self.coarse_RGBLoss(coarse_img, scaled_gt_x, False)
					if self.args.mode == 'xs2xs':
					   loss_dict['ce_loss'] = self.args.ce_weight*self.SegLoss(seg, torch.argmax(gt_seg, dim=1))   
				else:
					loss_dict = OrderedDict()

			else:
				loss_dict = OrderedDict()
				output_imgs = []
				output_inpaints = []
				output_masks = []
				output_segs = []
				last_rgb_output = torch.cat([data['frame1'], data['frame2']], dim=1).cuda(self.args.rank, non_blocking=True)
				last_seg_output = torch.cat([data['seg1'], data['seg2']], dim=1).cuda(self.args.rank, non_blocking=True)
				if self.args.num_pred_step > 1:
					assert self.args.num_pred_once == 1
				for i in range(self.args.num_pred_step):
					gt_start_frame_ind = 3+i*self.args.num_pred_once
					gt_x = torch.cat([data['frame'+str(i)] 
										 for i in range(gt_start_frame_ind, gt_start_frame_ind+self.args.num_pred_once)], dim=1)\
												.cuda(self.args.rank, non_blocking=True)
					gt_seg = gt_rgb = torch.cat([data['seg'+str(i)] 
										 for i in range(gt_start_frame_ind, gt_start_frame_ind+self.args.num_pred_once)], dim=1)\
												.cuda(self.args.rank, non_blocking=True)

					x = last_rgb_output
					seg = last_seg_output
					if self.args.fix_init_frames:
						x = torch.cat([data['frame2'].detach().cuda(self.args.rank, non_blocking=True), x], dim=1)
						seg = torch.cat([data['seg2'].detach().cuda(self.args.rank, non_blocking=True), seg], dim=1)

					coarse_img, refined_imgs, out_seg, attn_flow, mask, inpainted_img = self.model(x, seg=seg, gt_x=gt_x, gt_seg=gt_seg)
					for j in range(self.args.num_pred_once):
						prefix='step_{}_frame_{}'.format(i,j)
						loss_dict.update(self.coarse_RGBLoss(coarse_img[:,j*3:j*3+3], gt_x[:,j*3:j*3+3], False, prefix=prefix))
						loss_dict[prefix+'_ce_loss'] = self.args.ce_weight*self.SegLoss(out_seg[:,j*20:j*20+20], torch.argmax(gt_seg[:,j*20:j*20+20], dim=1))   
						output_imgs.append(coarse_img[:,j*3:j*3+3])
						output_segs.append(out_seg[:,j*20:j*20+20])

						if self.args.inpaint:
							prefix='step_{}_frame_{}_inpaint'.format(i,j)
							loss_dict.update(self.coarse_RGBLoss(inpainted_img[:,j*3:j*3+3]*(1-mask[:,j:j+1]), gt_x[:,j*3:j*3+3]*(1-mask[:,j:j+1]), False, prefix=prefix))
							loss_dict[prefix+'_mask_loss'] = 80*mask[:,j:j+1].mean()
							output_masks.append(mask[:,j:j+1])
							output_inpaints.append(inpainted_img[:,j*3:j*3+3])


					# following will only matter when num_pred_once == 1
					if self.args.num_pred_step == 1:
						break
					back_img = inpainted_img if self.args.inpaint else coarse_img
					last_rgb_output = torch.cat( [ x[:,-3:], back_img ], dim=1) 
					last_seg_output = torch.cat( [ seg[:,-20:], 
													torch.eye(20)[out_seg.argmax(dim=1)].permute(0,3,1,2).contiguous().cuda(self.args.rank, non_blocking=True)], dim=1)
			
			# loss and accuracy
			loss = 0
			for i in loss_dict.values():
				loss += torch.mean(i)
			loss_dict['loss_all'] = loss
			
			self.sync(loss_dict)

			# backward pass
			self.coarse_optimizer.zero_grad()
			self.refine_optimizer.zero_grad()

			loss_dict['loss_all'].backward()
			# init step in the first train steps
			if not self.args.lock_coarse and not self.args.lock_low:
				self.coarse_optimizer.step()
			if not self.args.lock_low and not self.args.lock_refine:
				self.refine_optimizer.step()
			comp_time += time() - end
			end = time()

			step_loss_record_dict = self.update_loss_record_dict(step_loss_record_dict, loss_dict, batch_size_p)


			if self.args.rank == 0:
				# add info to tensorboard
				info = {key:value.item() for key,value in loss_dict.items()}
				self.writer.add_scalars("losses", info, self.global_step)
				# print
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
						log = '\n\tframe{ind:.0f} l1 [{l1:.3f}] vgg [{vgg:.3f}] ssim [{ssim:.3f}] gdl [{gdl:.3f}] rgb_all [{rgb_all:.3f}] ce [{ce:.3f}]'.format(
								ind=i+1, 
								l1=step_loss_record_dict['step_frame_{}_coarse_l1_loss_record'.format(i+1)],
								vgg=step_loss_record_dict['step_frame_{}_coarse_vgg_loss_record'.format(i+1)],
								ssim=step_loss_record_dict['step_frame_{}_coarse_ssim_loss_record'.format(i+1)],
								gdl=step_loss_record_dict['step_frame_{}_coarse_gdl_loss_record'.format(i+1)],
								ce=step_loss_record_dict['step_frame_{}_coarse_ce_loss_record'.format(i+1)],
								rgb_all=step_loss_record_dict['step_frame_{}_coarse_all_loss_record'.format(i+1)]
							)
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

					for key,value in step_loss_record_dict.items():
						step_loss_record_dict[key] = 0

				if self.step % 30 == 0: 
					# image_set = self.prepare_image_set(data, coarse_img.cpu(), [ refined_img.cpu() for refined_img in refined_imgs], seg.cpu() if self.args.mode=='xs2xs' else None, flow=attn_flow)
					image_set = self.prepare_extra_image_set(data, [img.cpu() for img in output_imgs], [seg.cpu() for seg in output_segs], 
															[m.cpu() for m in output_masks] if self.args.inpaint else None,
															[m.cpu() for m in output_inpaints] if self.args.inpaint else None )
					self.writer.add_image('image_{}'.format(self.global_step), image_set, self.global_step)
		
		for key, value in step_loss_record_dict.items():
			epoch_key = key.replace('step', 'epoch')
			epoch_loss_record_dict[epoch_key]+=value

		if epoch_loss_record_dict['epoch_data_cnt'] != 0:
			for key, value in epoch_loss_record_dict.items():
				if key!='epoch_data_cnt':
					epoch_loss_record_dict[key] /= epoch_loss_record_dict['epoch_data_cnt']

		if self.args.rank == 0:
			log_main = 'Epoch [{epoch:d}/{tot_epoch:d}]'.format(epoch=self.epoch, tot_epoch=self.args.epochs)

			for i in range(self.args.num_pred_once*self.args.num_pred_step):
				log = '\n\tframe {ind:.0f} l1 [{l1:.3f}] vgg [{vgg:.3f}] ssim [{ssim:.3f}] gdl [{gdl:.3f}] rgb_all [{rgb_all:.3f}] ce [{ce:.3f}]'.format(
						ind=1+i, 
						l1=epoch_loss_record_dict['epoch_frame_{}_coarse_l1_loss_record'.format(i+1)],
						vgg=epoch_loss_record_dict['epoch_frame_{}_coarse_vgg_loss_record'.format(i+1)],
						ssim=epoch_loss_record_dict['epoch_frame_{}_coarse_ssim_loss_record'.format(i+1)],
						gdl=epoch_loss_record_dict['epoch_frame_{}_coarse_gdl_loss_record'.format(i+1)],
						ce=epoch_loss_record_dict['epoch_frame_{}_coarse_ce_loss_record'.format(i+1)],
						rgb_all=epoch_loss_record_dict['epoch_frame_{}_coarse_all_loss_record'.format(i+1)]
					)
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

	def validate(self, val_coarse=True):
		self.args.logger.info('Validation epoch {} started'.format(self.epoch))
		self.model.eval()

		val_criteria = {}

		for i in range(self.args.vid_length):
			val_criteria['{:d}_refine_l1'.format(i)]    = AverageMeter()
			val_criteria['{:d}_refine_psnr'.format(i)]  = AverageMeter()
			val_criteria['{:d}_refine_ssim'.format(i)]  = AverageMeter()
			val_criteria['{:d}_refine_vgg'.format(i)]   = AverageMeter()
			if val_coarse:
				val_criteria['{:d}_l1'.format(i)] = AverageMeter()
				val_criteria['{:d}_psnr'.format(i)] = AverageMeter()
				val_criteria['{:d}_ssim'.format(i)] = AverageMeter()
				if self.args.mode == 'xs2xs':
					val_criteria['{:d}_iou'.format(i)] = AverageMeter()
				val_criteria['{:d}_vgg'.format(i)] = AverageMeter()
			if self.args.high_res:
				val_criteria['{:d}_h_l1'.format(i)]  = AverageMeter()
				val_criteria['{:d}_h_psnr'.format(i)] = AverageMeter()
				val_criteria['{:d}_h_ssim'.format(i)] = AverageMeter()
				val_criteria['{:d}_h_vgg'.format(i)]  = AverageMeter()
			if self.args.re_ref:
				val_criteria['{:d}_re_ref_l1'.format(i)]  = AverageMeter()
				val_criteria['{:d}_re_ref_psnr'.format(i)] = AverageMeter()
				val_criteria['{:d}_re_ref_ssim'.format(i)] = AverageMeter()
				val_criteria['{:d}_re_ref_vgg'.format(i)]  = AverageMeter()
		step_losses = OrderedDict()

		with torch.no_grad():
			end = time()
			load_time = 0
			comp_time = 0
			for i, data in enumerate(self.val_loader):
				load_time += time()-end
				end = time()
				self.step=i

				start_x, seg, fg_mask, gt_xs, gt_segs = self.get_inputs(data)
				show_high_res_imgs=None
				show_re_ref_imgs=None
				show_segs = None

				show_coarse_imgs = []
				show_refine_imgs = []
				if self.args.mode == 'xs2xs':
					show_segs = []
				if self.args.high_res:
					show_high_res_imgs = []
				if self.args.re_ref:
					show_re_ref_imgs = []
				x=start_x
				for pred_step in range(self.args.vid_length):
					gt_x = gt_xs[pred_step]
					gt_seg = gt_segs[pred_step]

					size = x.size(0)
					x = x.cuda(self.args.rank, non_blocking=True)
					seg = seg.cuda(self.args.rank, non_blocking=True)  if self.args.mode == 'xs2xs' else None
					fg_mask = None#fg_mask.cuda(self.args.rank, non_blocking=True)  if self.args.mode == 'xs2xs' else None
					gt_x = gt_x.cuda(self.args.rank, non_blocking=True)
					gt_seg = gt_seg.cuda(self.args.rank, non_blocking=True)  if self.args.mode == 'xs2xs' else None

					coarse_img, refine_imgs, refined_h_img, re_ref_img, out_seg, attn_flow = self.model(x, fg_mask, seg=seg, gt_x=gt_x, gt_seg=gt_seg)
					coarse_gt = F.interpolate(gt_x, scale_factor=0.5, mode='bilinear', align_corners=True) if self.args.high_res else gt_x
					refine_imgs = [refine_imgs[i].clamp_(-1, 1) for i in range(len(refine_imgs))]
					show_coarse_imgs.append(coarse_img.cpu())
					show_refine_imgs.append(refine_imgs[-1].cpu())
					show_segs.append(out_seg.cpu())

					
					# rgb criteria
					if val_coarse:
						step_losses['{:d}_l1'.format(pred_step)] =   self.L1Loss(coarse_img, coarse_gt)
						step_losses['{:d}_psnr'.format(pred_step)] = self.PSNRLoss((coarse_img+1)/2, (coarse_gt+1)/2)
						step_losses['{:d}_ssim'.format(pred_step)] = 1-self.SSIMLoss(coarse_img, coarse_gt)
						if self.args.mode == 'xs2xs':
							step_losses['{:d}_iou'.format(pred_step)] =  self.IoULoss(torch.argmax(out_seg, dim=1), torch.argmax(gt_seg, dim=1))
						step_losses['{:d}_vgg'.format(pred_step)] =  self.VGGCosLoss(coarse_img, coarse_gt, False)

					if self.args.mode == 'xs2xs':
						out_seg = torch.eye(20)[out_seg.argmax(dim=1)].permute(0,3,1,2).contiguous().cuda(self.args.rank, non_blocking=True)
						seg = torch.cat([seg[:,20:], out_seg], dim=1)

					step_losses['{:d}_refine_l1'.format(pred_step)] =   self.L1Loss(refine_imgs[-1], coarse_gt)
					step_losses['{:d}_refine_psnr'.format(pred_step)] = self.PSNRLoss((refine_imgs[-1]+1)/2, (coarse_gt+1)/2)
					step_losses['{:d}_refine_ssim'.format(pred_step)] = 1-self.SSIMLoss(refine_imgs[-1], coarse_gt)
					step_losses['{:d}_refine_vgg'.format(pred_step)] =  self.VGGCosLoss(refine_imgs[-1], coarse_gt, False)

					if self.args.high_res:
						refined_h_img = refined_h_img.clamp_(-1, 1)
						show_high_res_imgs.append(refined_h_img)
						step_losses['{:d}_h_l1'.format(pred_step)] =   self.L1Loss(refined_h_img, gt_x)
						step_losses['{:d}_h_psnr'.format(pred_step)] = self.PSNRLoss((refined_h_img+1)/2, (gt_x+1)/2)
						step_losses['{:d}_h_ssim'.format(pred_step)] = 1-self.SSIMLoss(refined_h_img, gt_x)
						step_losses['{:d}_h_vgg'.format(pred_step)] =  self.VGGCosLoss(refined_h_img, gt_x, False)

					if self.args.re_ref:
						re_ref_img = re_ref_img.clamp_(-1, 1)
						show_re_ref_imgs.append(re_ref_img.cpu())
						step_losses['{:d}_re_ref_l1'.format(pred_step)] =   self.L1Loss(re_ref_img, gt_x)
						step_losses['{:d}_re_ref_psnr'.format(pred_step)] = self.PSNRLoss((re_ref_img+1)/2, (gt_x+1)/2)
						step_losses['{:d}_re_ref_ssim'.format(pred_step)] = 1-self.SSIMLoss(re_ref_img, gt_x)
						step_losses['{:d}_re_ref_vgg'.format(pred_step)] =  self.VGGCosLoss(re_ref_img, gt_x, False)
						x = torch.cat([x[:, 3:], re_ref_img], dim=1)
					else:
						x = torch.cat([x[:,3:], refine_imgs[-1]], dim=1)

				self.sync(step_losses) # sum
				for key in list(val_criteria.keys()):
					val_criteria[key].update(step_losses[key].cpu().item(), size*self.args.gpus)



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
					if self.step % 1 == 0:
						if self.args.high_res:
							image_set = self.prepare_image_sets(data, show_coarse_imgs, show_refine_imgs, show_segs, high_res_img=show_high_res_imgs)
						elif self.args.re_ref:
							image_set = self.prepare_image_sets(data, show_coarse_imgs, show_refine_imgs, show_segs, re_ref_imgs=show_re_ref_imgs)
						else:
							image_set = self.prepare_image_sets(data, show_coarse_imgs, show_refine_imgs, show_segs)
						image_name = 'e{}_img_{}'.format(self.epoch, self.step)
						self.writer.add_image(image_name, image_set, self.step)

		if self.args.rank == 0:
			logss=""
			for i in range(self.args.vid_length):
				logs ="pred {:d}".format(i)
				if val_coarse:
					coarse_logs = "\tcoarse L1: {l1:.3f}\tPSNR: {psnr:.3f}\tSSIM: {ssim:.3f}\tvgg: {vgg:.3f}\t".format(
							l1=val_criteria['{:d}_l1'.format(i)].avg,
							psnr=val_criteria['{:d}_psnr'.format(i)].avg,
							ssim=val_criteria['{:d}_ssim'.format(i)].avg,
							vgg = val_criteria['{:d}_vgg'.format(i)].avg,
							iou = val_criteria['{:d}_iou'.format(i)].avg
					)
					if self.args.mode=='xs2xs':
						coarse_logs+='iou: {iou:.3f}\n'.format(iou = val_criteria['{:d}_iou'.format(i)].avg)
					logs += coarse_logs

				logs += "\trefine L1: {r_l1:.3f}\tPSNR: {r_psnr:.3f}\tSSIM: {r_ssim:.3f}\tvgg: {r_vgg:.3f}\n".format(
							r_l1=val_criteria['{:d}_refine_l1'.format(i)].avg,
							r_psnr=val_criteria['{:d}_refine_psnr'.format(i)].avg,
							r_ssim=val_criteria['{:d}_refine_ssim'.format(i)].avg,
							r_vgg = val_criteria['{:d}_refine_vgg'.format(i)].avg
					)


				if self.args.high_res:
					highres_logs = "\thighre L1: {h_l1:.3f}\tPSNR: {h_psnr:.3f}\tSSIM: {h_ssim:.3f}\tvgg: {h_vgg:.3f}\n".format(
							h_l1=val_criteria['{:d}_h_l1'.format(i)].avg,
							h_psnr=val_criteria['{:d}_h_psnr'.format(i)].avg,
							h_ssim=val_criteria['{:d}_h_ssim'.format(i)].avg,
							h_vgg = val_criteria['{:d}_h_vgg'.format(i)].avg
					)
					logs += highres_logs

				if self.args.re_ref:
					re_ref_logs = "\tre_ref  L1: {h_l1:.3f}\tPSNR: {h_psnr:.3f}\tSSIM: {h_ssim:.3f}\tvgg: {h_vgg:.3f}\n".format(
							h_l1=val_criteria['{:d}_re_ref_l1'.format(i)].avg,
							h_psnr=val_criteria['{:d}_re_ref_psnr'.format(i)].avg,
							h_ssim=val_criteria['{:d}_re_ref_ssim'.format(i)].avg,
							h_vgg = val_criteria['{:d}_re_ref_vgg'.format(i)].avg
					)
					logs += re_ref_logs
				logss+=logs

			logs = "Epoch [{epoch:d}]\n".format(epoch=self.epoch) + logss
			self.args.logger.info(logs)

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

					save_image(pred_img, '{}/{}_pred.png'.format(self.args.imgout_dir, img_count))
					save_image(gt_img, '{}/{}_gt.png'.format(self.args.imgout_dir, img_count))
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

	def mycycgen(self):
		assert self.args.rank == 0 # only allow single worker
		if self.args.syn_type == 'extra':
			load_dir_split = 'extra_wing'
			seg_load_dir_split = 'extra_wing/seg'
			save_dir_split = 'extra_wing'#.format(int(self.args.interval), self.args.vid_length)
		elif self.args.syn_type == 'inter':
			load_dir_split = 'extra_wing'
			seg_load_dir_split = 'extra_wing/seg'
			save_dir_split = 'extra_wing'
		with open('/data/linz/proj/Dataset/Cityscape/load_files/root_clip.pkl', 'rb') as f:
			clips = pickle.load(f)
			clips_dir = clips['val'][:61] # onlye generate 0-60

		load_img_dir = 'cycgen/cityscape/{}x{}/{}'.format(self.args.input_h, self.args.input_w, load_dir_split)
		save_img_dir = 'cycgen/cityscape/{}x{}/{}'.format(self.args.input_h, self.args.input_w, save_dir_split)
		load_clip_dirs = [load_img_dir+'/'+clip_dir[0] for clip_dir in clips_dir]
		if self.args.mode == 'xs2xs':
			load_seg_dir = 'cycgen/cityscape/{}x{}/{}'.format(self.args.input_h, self.args.input_w, seg_load_dir_split)
			save_seg_dir = 'cycgen/cityscape/{}x{}/{}/seg'.format(self.args.input_h, self.args.input_w, save_dir_split)
			load_clip_seg_dirs = [load_seg_dir+'/'+clip_dir[0] for clip_dir in clips_dir]

		for first_index in [25]:

			# first_index   = 10
			second_index    = first_index+2
			# pred_index = 3

			pred_index_name = first_index+1

			end = time()
			for clip_ind, load_clip_dir in enumerate(load_clip_dirs):
				load_img_files = glob.glob(load_clip_dir+"/*.png")
				load_img_files = [  load_clip_dir+'/{:0>2d}.0.png'.format(first_index),
									load_clip_dir+'/{:0>2d}.0.png'.format(second_index) ]
				# load_img_files.sort()
				load_imgs = rgb_load(load_img_files)

				if self.args.mode == 'xs2xs':
					# load_seg_files = glob.glob(load_clip_seg_dirs[clip_ind]+"/*.png")
					# load_seg_files.sort()
					load_seg_files = [  load_clip_seg_dirs[clip_ind]+'/{:0>2d}.0.png'.format(first_index),
										load_clip_seg_dirs[clip_ind]+'/{:0>2d}.0.png'.format(second_index) ]
					load_segs = seg_load(load_seg_files)
					load_segs = [ np.eye(20)[np.array(i)] for i in load_segs ] 

				for i in range(len(load_imgs)):
					load_imgs[i] = transforms.functional.normalize( 
											transforms.functional.to_tensor(
												load_imgs[i]
											),  (0.5, 0.5, 0.5),(0.5, 0.5, 0.5)
										).unsqueeze(0)

					load_segs[i] = torch.from_numpy(
												np.transpose(
													load_segs[i], (2,0,1)
													)
												).float().unsqueeze(0)

				# if self.args.syn_type=='inter':
				#   # load data
				#   input_imgs = load_imgs
				#   input_len = len(input_imgs)
				#   pred_img_name_prefix = '/'.join(load_img_files[0].split('/')[-4:-1])
				#   save_img_prefix = save_img_dir + '/' + pred_img_name_prefix
				#   if not os.path.exists(save_img_prefix):
				#       os.makedirs(save_img_prefix)
				#   for step in range(input_len-1):
				#       prev_img_name = '/'.join(load_img_files[step].split('/')[-4:])
				#       prev_img_ind  = prev_img_name.split('/')[-1][:-4]
				#       pred_img_ind = str(float(prev_img_ind) + self.args.interval/2)
				#       pred_img_ind_split = pred_img_ind.split('.')
				#       pred_img_ind_int = '{:0>2d}'.format(int(pred_img_ind_split[0]))
				#       pred_img_ind_split[0] = pred_img_ind_int
				#       pred_img_ind = '.'.join(pred_img_ind_split)
				#       pred_img_name = save_img_prefix + '/' + pred_img_ind+".png"

				#       x = torch.cat(input_imgs[step:step+2], dim=1)
				#       x = x.cuda(self.args.rank, non_blocking=True)
				#       seg = None
				#       fg_mask = None
				#       gt_x = None
				#       gt_seg =  None

				#       coarse_img, refine_imgs, refined_h_img, seg, attn_flow = self.model(x, fg_mask, seg=seg, gt_x=gt_x, gt_seg=gt_seg)
				#       pred_img = refine_imgs[-1][0]
				#       pred_img = self.normalize(pred_img)

				#       save_image(pred_img, pred_img_name)
				#   # save gt
				#   for step in range(input_len):
				#       load_img_name = '/'.join(load_img_files[step].split('/')[-4:])
				#       save_img_name = save_img_dir + '/' + load_img_name
				#       shutil.copyfile(load_img_files[step], save_img_name)

				# elif self.args.syn_type == 'extra':
				# load data
				input_imgs = load_imgs #[ load_imgs[first_index],load_imgs[second_index] ] 

				input_imgs = [t.cuda(self.args.rank) for t in input_imgs]
				# img1_ind = float(load_img_files[0].split('/')[-1][:-4])
				# img2_ind = float(load_img_files[int(self.args.interval)].split('/')[-1][:-4])
				# time_interval = img2_ind - img1_ind
				pred_img_name_prefix = '/'.join(load_img_files[0].split('/')[-4:-1])
				save_img_prefix = save_img_dir + '/' + pred_img_name_prefix
				if not os.path.exists(save_img_prefix):
					os.makedirs(save_img_prefix)

				# load seg
				input_segs = load_segs #[ load_segs[first_index], load_segs[second_index] ]
				input_segs = [t.cuda(self.args.rank) for t in input_segs]
				pred_seg_name_prefix = '/'.join(load_seg_files[0].split('/')[-4:-1])
				save_seg_prefix = save_seg_dir + '/' + pred_seg_name_prefix
				if not os.path.exists(save_seg_prefix):
					os.makedirs(save_seg_prefix)

				# for step in range(self.args.vid_length):
				input_segs = [t.cuda(self.args.rank) for t in input_segs]
				pred_img_ind = pred_index_name #pred_img2_ind + (step+1)*time_interval
				# following 5 lines make integer part _ _ 
				pred_img_ind = str(pred_img_ind)
				pred_img_ind_split = pred_img_ind.split('.')
				pred_img_ind_int = '{:0>2d}.0'.format(int(pred_img_ind_split[0]))
				pred_img_ind_split[0] = pred_img_ind_int
				pred_img_ind = '.'.join(pred_img_ind_split)                 

				pred_img_name = save_img_prefix + '/' + pred_img_ind+".png"
				pred_seg_name = save_seg_prefix + '/' + pred_img_ind+".png"

				x = torch.cat(input_imgs, dim=1)
				x = x.cuda(self.args.rank, non_blocking=True)
				seg = torch.cat(input_segs, dim=1)
				fg_mask = None
				gt_x = None
				gt_seg =  None

				coarse_img, refine_imgs, refined_h_img, re_ref_img, out_seg, attn_flow = self.model(x, fg_mask, seg=seg, gt_x=gt_x, gt_seg=gt_seg)
				if self.args.re_ref:
					pred_img = re_ref_img
				else:
					pred_img = refine_imgs[-1]
				# input_imgs = [input_imgs[1]] + [pred_img]
				# input_segs = [input_segs[1]] + [F.softmax(out_seg, dim=1)]
				save_image(out_seg.argmax(dim=1)[0].float()/255, pred_seg_name)

				out_seg = torch.eye(20)[out_seg.argmax(dim=1)].permute(0,3,1,2).contiguous()
				# input_segs = [input_segs[1]] + [out_seg]

				pred_img = pred_img[0]
				pred_img = self.normalize(pred_img)
				save_image(pred_img, pred_img_name)

				

				# vis_out_seg = vis_seg_mask(out_seg, 20)
				# pred_seg = vis_out_seg[0]
				# save_image(pred_seg, pred_seg_name)

				# save gt
				# for step in range(2):
				# load_img_name = '/'.join(load_img_files[first_index].split('/')[-4:])
				# save_img_name = save_img_dir + '/' + load_img_name
				# shutil.copyfile(load_img_files[first_index], save_img_name)
				# if self.args.mode == 'xs2xs':
				#   load_seg_name = '/'.join(load_seg_files[first_index].split('/')[-4:])
				#   save_seg_name = save_seg_dir + '/' + load_seg_name
				#   # vis_out_seg = vis_seg_mask(load_segs[first_index], 20)[0]
				#   shutil.copyfile(load_seg_files[first_index], save_seg_name)

				# load_img_name = '/'.join(load_img_files[second_index].split('/')[-4:])
				# save_img_name = save_img_dir + '/' + load_img_name
				# shutil.copyfile(load_img_files[second_index], save_img_name)
				# if self.args.mode == 'xs2xs':
				#   load_seg_name = '/'.join(load_seg_files[second_index].split('/')[-4:])
				#   save_seg_name = save_seg_dir + '/' + load_seg_name
				#   # vis_out_seg = vis_seg_mask(load_segs[second_index], 20)[0]
				#   shutil.copyfile(load_seg_files[second_index], save_seg_name)

				p_time = time() - end
				end = time()
				sys.stdout.write('\rprocessing {}/{} {}s {}'.format(clip_ind, 61, p_time, load_img_files[0]))           




	def cycgen(self):
		assert self.args.rank == 0 # only allow single worker
		if self.args.syn_type == 'extra':
			load_dir_split = 'gt'
			seg_load_dir_split = 'gt_seg'
			save_dir_split = 'extra_{}x{}'.format(int(self.args.interval), self.args.vid_length)
		elif self.args.syn_type == 'inter':
			save_dir_split = 'inter_x{:d}'.format(int(2/self.args.interval))
			load_dir_split = 'inter_x{:d}'.format(int(1/self.args.interval)) if self.args.interval<1 else 'gt'
		with open('/data/linz/proj/Dataset/Cityscape/load_files/root_clip.pkl', 'rb') as f:
			clips = pickle.load(f)
			clips_dir = clips['val'][:61] # onlye generate 0-60

		load_img_dir = 'cycgen/cityscape/{}x{}/{}'.format(self.args.input_h, self.args.input_w, load_dir_split)
		save_img_dir = 'cycgen/cityscape/{}x{}/{}'.format(self.args.input_h, self.args.input_w, save_dir_split)
		load_clip_dirs = [load_img_dir+'/'+clip_dir[0] for clip_dir in clips_dir]
		if self.args.mode == 'xs2xs':
			load_seg_dir = 'cycgen/cityscape/{}x{}/{}'.format(self.args.input_h, self.args.input_w, seg_load_dir_split)
			save_seg_dir = 'cycgen/cityscape/{}x{}/{}/seg'.format(self.args.input_h, self.args.input_w, save_dir_split)
			load_clip_seg_dirs = [load_seg_dir+'/'+clip_dir[0] for clip_dir in clips_dir]

		end = time()
		for clip_ind, load_clip_dir in enumerate(load_clip_dirs):
			load_img_files = glob.glob(load_clip_dir+"/*.png")
			load_img_files.sort()
			load_imgs = rgb_load(load_img_files)

			if self.args.mode == 'xs2xs':
				load_seg_files = glob.glob(load_clip_seg_dirs[clip_ind]+"/*.png")
				load_seg_files.sort()
				load_segs = seg_load(load_seg_files)
				load_segs = [ np.eye(20)[np.array(i)] for i in load_segs ] 

			for i in range(len(load_imgs)):
				load_imgs[i] = transforms.functional.normalize( 
										transforms.functional.to_tensor(
											load_imgs[i]
										),  (0.5, 0.5, 0.5),(0.5, 0.5, 0.5)
									).unsqueeze(0)

				load_segs[i] = torch.from_numpy(
											np.transpose(
												load_segs[i], (2,0,1)
												)
											).float().unsqueeze(0)

			if self.args.syn_type=='inter':
				# load data
				input_imgs = load_imgs
				input_len = len(input_imgs)
				pred_img_name_prefix = '/'.join(load_img_files[0].split('/')[-4:-1])
				save_img_prefix = save_img_dir + '/' + pred_img_name_prefix
				if not os.path.exists(save_img_prefix):
					os.makedirs(save_img_prefix)
				for step in range(input_len-1):
					prev_img_name = '/'.join(load_img_files[step].split('/')[-4:])
					prev_img_ind  = prev_img_name.split('/')[-1][:-4]
					pred_img_ind = str(float(prev_img_ind) + self.args.interval/2)
					pred_img_ind_split = pred_img_ind.split('.')
					pred_img_ind_int = '{:0>2d}'.format(int(pred_img_ind_split[0]))
					pred_img_ind_split[0] = pred_img_ind_int
					pred_img_ind = '.'.join(pred_img_ind_split)
					pred_img_name = save_img_prefix + '/' + pred_img_ind+".png"

					x = torch.cat(input_imgs[step:step+2], dim=1)
					x = x.cuda(self.args.rank, non_blocking=True)
					seg = None
					fg_mask = None
					gt_x = None
					gt_seg =  None

					coarse_img, refine_imgs, refined_h_img, seg, attn_flow = self.model(x, fg_mask, seg=seg, gt_x=gt_x, gt_seg=gt_seg)
					pred_img = refine_imgs[-1][0]
					pred_img = self.normalize(pred_img)

					save_image(pred_img, pred_img_name)
				# save gt
				for step in range(input_len):
					load_img_name = '/'.join(load_img_files[step].split('/')[-4:])
					save_img_name = save_img_dir + '/' + load_img_name
					shutil.copyfile(load_img_files[step], save_img_name)

			elif self.args.syn_type == 'extra':
				# load data
				input_imgs = load_imgs[:2*int(self.args.interval):int(self.args.interval)]
				input_imgs = [t.cuda(self.args.rank) for t in input_imgs]
				img1_ind = float(load_img_files[0].split('/')[-1][:-4])
				img2_ind = float(load_img_files[int(self.args.interval)].split('/')[-1][:-4])
				time_interval = img2_ind - img1_ind
				pred_img_name_prefix = '/'.join(load_img_files[0].split('/')[-4:-1])
				save_img_prefix = save_img_dir + '/' + pred_img_name_prefix
				if not os.path.exists(save_img_prefix):
					os.makedirs(save_img_prefix)

				# load seg
				input_segs = load_segs[:2*int(self.args.interval):int(self.args.interval)]
				input_segs = [t.cuda(self.args.rank) for t in input_segs]
				pred_seg_name_prefix = '/'.join(load_seg_files[0].split('/')[-4:-1])
				save_seg_prefix = save_seg_dir + '/' + pred_seg_name_prefix
				if not os.path.exists(save_seg_prefix):
					os.makedirs(save_seg_prefix)

				for step in range(self.args.vid_length):
					input_segs = [t.cuda(self.args.rank) for t in input_segs]
					pred_img_ind = img2_ind + (step+1)*time_interval
					# following 5 lines make integer part _ _ 
					pred_img_ind = str(pred_img_ind)
					pred_img_ind_split = pred_img_ind.split('.')
					pred_img_ind_int = '{:0>2d}'.format(int(pred_img_ind_split[0]))
					pred_img_ind_split[0] = pred_img_ind_int
					pred_img_ind = '.'.join(pred_img_ind_split)                 

					pred_img_name = save_img_prefix + '/' + pred_img_ind+".png"
					pred_seg_name = save_seg_prefix + '/' + pred_img_ind+".png"

					x = torch.cat(input_imgs, dim=1)
					x = x.cuda(self.args.rank, non_blocking=True)
					seg = torch.cat(input_segs, dim=1)
					fg_mask = None
					gt_x = None
					gt_seg =  None

					coarse_img, refine_imgs, refined_h_img, re_ref_img, out_seg, attn_flow = self.model(x, fg_mask, seg=seg, gt_x=gt_x, gt_seg=gt_seg)
					if self.args.re_ref:
						pred_img = re_ref_img
					else:
						pred_img = refine_imgs[-1]
					input_imgs = [input_imgs[1]] + [pred_img]
					input_segs = [input_segs[1]] + [F.softmax(out_seg, dim=1)]
					out_seg = torch.eye(20)[out_seg.argmax(dim=1)].permute(0,3,1,2).contiguous()
					# input_segs = [input_segs[1]] + [out_seg]

					pred_img = pred_img[0]
					pred_img = self.normalize(pred_img)
					save_image(pred_img, pred_img_name)

					vis_out_seg = vis_seg_mask(out_seg, 20)
					pred_seg = vis_out_seg[0]
					save_image(pred_seg, pred_seg_name)

				# save gt
				for step in range(2):
					load_img_name = '/'.join(load_img_files[step*int(self.args.interval)].split('/')[-4:])
					save_img_name = save_img_dir + '/' + load_img_name
					shutil.copyfile(load_img_files[step*int(self.args.interval)], save_img_name)
					if self.args.mode == 'xs2xs':
						load_seg_name = '/'.join(load_seg_files[step*int(self.args.interval)].split('/')[-4:])
						save_seg_name = save_seg_dir + '/' + load_seg_name
						vis_out_seg = vis_seg_mask(load_segs[step*int(self.args.interval)], 20)[0]
						save_image(vis_out_seg, save_seg_name)

			p_time = time() - end
			end = time()
			sys.stdout.write('\rprocessing {}/{} {}s {}'.format(clip_ind, 61, p_time, load_img_files[0]))           

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
			'model': self.model.module.state_dict(),
			'coarse_optimizer': self.coarse_optimizer.state_dict(),
			'refine_optimizer': self.refine_optimizer.state_dict()
		}
		if self.args.high_res:
			save_dict['high_res_optimizer'] = self.high_res_optimizer.state_dict()
		if self.args.re_ref:
			save_dict['re_ref_optimizer'] = self.re_ref_optimizer.state_dict()
		torch.save(save_dict, save_name)
		self.args.logger.info('save model: {}'.format(save_name))

	def load_checkpoint(self):
		# load_md_dir = '{}_{}_{}_{}'.format(self.args.model, self.args.mode, self.args.syn_type, self.args.checksession) 
		load_md_dir = '{}_{}_{}_{}'.format(self.args.model, self.args.mode, self.args.syn_type, self.args.checksession) 
		if self.args.load_dir is not None:
			load_name = os.path.join(self.args.load_dir,
									'checkpoint',
									load_md_dir+'_{}_{}.pth'.format(self.args.checkepoch, self.args.checkpoint))
		else:
			load_name = os.path.join(load_md_dir+'_{}_{}.pth'.format(self.args.checkepoch, self.args.checkpoint))
		self.args.logger.info('Loading checkpoint %s' % load_name)
		ckpt = torch.load(load_name)

		if self.args.seperate_coarse:
			load_md_dir = '{}_{}_{}_{}'.format(self.args.model, self.args.coarse_mode, self.args.syn_type, self.args.coarse_checksession) 
			load_name = os.path.join(self.args.coarse_load_dir,
									'checkpoint',
									load_md_dir+'_{}_{}.pth'.format(self.args.coarse_checkepoch, self.args.coarse_checkpoint))
			self.args.logger.info('Loading coarse checkpoint %s' % load_name)
			coarse_ckpt = torch.load(load_name)



		# transer model parameters
		if not self.args.seperate_coarse:
			if not self.args.resume and (self.args.lock_coarse or self.args.pretrained_coarse):
				model_dict = self.model.module.state_dict()
				new_ckpt = OrderedDict()
				for key,item in ckpt['model'].items():
					if 'coarse' in key:
						new_ckpt[key] = item
				model_dict.update(new_ckpt)
				self.model.module.load_state_dict(model_dict)
			elif self.args.pretrained_low and not self.args.resume:
				model_dict = self.model.module.state_dict()
				new_ckpt = OrderedDict()
				for key,item in ckpt['model'].items():
					if 'coarse' in key or 'refine' in key:
						new_ckpt[key] = item
				model_dict.update(new_ckpt)
				self.model.module.load_state_dict(model_dict)
			else:
				self.model.module.load_state_dict(ckpt['model'])

		# load coarse model from coarse ckpt
		else:
			model_dict = self.model.module.state_dict()
			new_ckpt = OrderedDict()
			for key,item in coarse_ckpt['model'].items():
				if 'coarse' in key:
					new_ckpt[key] = item
			for key,item in ckpt['model'].items():
				if 'refine' in key:
					new_ckpt[key] = item
			model_dict.update(new_ckpt)
			self.model.module.load_state_dict(model_dict)

		# transfer opt params to current device
		if self.args.resume or (not self.args.lock_coarse and not ( self.args.lock_low and not self.args.resume) ) :
			if self.args.split == 'train':
				self.coarse_optimizer.load_state_dict(ckpt['coarse_optimizer'])
				for state in self.coarse_optimizer.state.values():
					for k, v in state.items():
						if isinstance(v, torch.Tensor):
							state[k] = v.cuda(self.args.rank) 
				if not self.args.pretrained_coarse: # pretrained low or whole model
					self.refine_optimizer.load_state_dict(ckpt['refine_optimizer'])
					for state in self.refine_optimizer.state.values():
						for k, v in state.items():
							if isinstance(v, torch.Tensor):
								state[k] = v.cuda(self.args.rank) 
					if not ( self.args.pretrained_low and not self.args.resume): # whole model 
						if self.args.high_res:
							self.high_res_optimizer.load_state_dict(ckpt['high_res_optimizer'])
							for state in self.high_res_optimizer.state.values():
								for k, v in state.items():
									if isinstance(v, torch.Tensor):
										state[k] = v.cuda(self.args.rank) 
						self.epoch = ckpt['epoch']
						self.global_step = (self.epoch-1)*len(self.train_loader)         
			else :
				assert ckpt['epoch']-1 == self.args.checkepoch, [ckpt['epoch'], self.args.checkepoch]
				self.epoch = ckpt['epoch'] - 1
		self.args.logger.info('checkpoint loaded')

