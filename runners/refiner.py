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
from folder import rgb_load
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
		if self.args.lock_low:
			for p in self.model.refine_model.parameters():
				p.requires_grad = False
		coarse_params_cnt = count_parameters(self.model.coarse_model)
		refine_params_cnt = count_parameters(self.model.refine_model)
		print("coarse params ", coarse_params_cnt)
		print("refine params ", refine_params_cnt)
		if self.args.re_ref:
			re_ref_params_cnt = count_parameters(self.model.re_ref_model)
			print("re_ref params ", re_ref_params_cnt)
		torch.cuda.set_device(args.rank)
		self.model.cuda(args.rank)
		self.model = torch.nn.parallel.DistributedDataParallel(self.model,
				device_ids=[args.rank])
		if self.args.split != 'cycgen':
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
			# if args.split:
			# 	self.writer =  SummaryWriter(args.path+'/val_int_{}_logs'.format(self.args.interval)) if self.args.dataset=='cityscape' else \
			# 							SummaryWriter(args.path+'/{}_val_int_1_logs'.format(self.args.dataset))
			# 	# SummaryWriter(args.path+'/val_logs') if args.interval == 2 else\
								

			# else:
			# 	self.writer = SummaryWriter(args.path+'/logs')
			writer_name = args.path+'/{}_int_{}_{}_logs'.format(self.args.split, self.args.interval, self.args.dataset)
			self.writer = SummaryWriter(writer_name)

		self.base_flow_map = self.standard_flow_map()
		self.base_flow_map = F.interpolate(torch.tensor(self.base_flow_map).permute(0,3,1,2), size=(128, 256), mode='bilinear', align_corners=True)[0]

	def set_epoch(self, epoch):
		self.args.logger.info("Start of epoch %d" % (epoch+1))
		self.epoch = epoch + 1
		self.train_loader.sampler.set_epoch(epoch)
		# self.val_loader.sampler.set_epoch(epoch)

	def get_input(self, data):
		# assert self.args.mode == 'xs2xs'
		# if not self.args.high_res:
		#   if self.args.syn_type == 'extra':
		#       x = torch.cat([data['frame1'], data['frame2'], data['seg1'], data['seg2']], dim=1)
		#       mask = torch.cat([data['fg_mask1'],data['fg_mask2']], dim=1)
		#       gt = torch.cat([data['frame3'], data['seg3']], dim=1)
		#   else:
		#       x = torch.cat([data['frame1'], data['frame3'], data['seg1'], data['seg3']], dim=1)
		#       mask = torch.cat([data['fg_mask1'],data['fg_mask3']], dim=1)
		#       gt = torch.cat([data['frame2'], data['seg2']], dim=1)   
		#   return x, mask, gt     
		# else:
		if self.args.syn_type == 'extra':
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

	def normalize(self, img):
		return (img+1)/2

	def standard_flow_map(self):
		offset = torch.zeros(1,2,16,32)
		h_unit = 16//5
		w_unit = 32//9
		for i in range(5):
			offset[:,1,i*h_unit:min((i+1)*h_unit, 16), :] = i
		for i in range(9):
			offset[:,0,:, i*w_unit:min((i+1)*w_unit, 32)] = i


		# offset[:,0,:10, :] = 0
		# offset[:,0,10:20, :] = 1
		# offset[:,0,20:32, :] = 2
		# offset[:,1,:, :20] = 0
		# offset[:,1,:, 20:40] = 1
		# offset[:,1,:, 40:64] = 2
		# h_add = torch.arange(32).view(1, 1, 32, 1).expand(1, 1, -1, 64).float()
		# w_add = torch.arange(64).view(1, 1, 1, 64).expand(1, 1, 32, -1).float()
		h_w_add = torch.zeros(1, 2, 16, 32)
		h_w_add[:, 0] = 4
		h_w_add[:, 1] = 2
		offset = offset - h_w_add
		flow_map = flow_to_image(flow=offset.permute(0,2,3,1).contiguous().numpy())
		return flow_map

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
		# refine
		refined_bs_imgs = [ refined_img[0].unsqueeze(0) for refined_img in refined_imgs ] 
		# for i in range(self.args.n_scales):
		insert_img = F.interpolate(refined_bs_imgs[-1], size=re_size)[0].clamp_(-1, 1) 

		pred_rgb = self.normalize(insert_img)
		insert_ind = insert_index + 1
		view_rgbs.insert(insert_ind, pred_rgb)


		view_imgs = view_rgbs 
		if self.args.mode=='xs2xs':
			view_imgs = view_imgs + view_segs
		

		if self.args.high_res:   
			view_imgs = view_rgbs[:2] + [ self.normalize(high_res_img[0].clamp_(-1,1))] + [view_rgbs[-1]] + \
						[black_img] + [F.interpolate(self.normalize(data['frame2'][0].unsqueeze(0)), scale_factor=0.5, mode='bilinear', align_corners=True)[0]] + view_rgbs[2:4] 

			if self.args.mode=='xs2xs':
				view_imgs += view_segs
			re_size = (256, 512)
			view_imgs = [F.interpolate(img.unsqueeze(0), size=(256, 512), mode='bilinear', align_corners=True)[0] for img in view_imgs]
			n_rows = 4
		elif self.args.re_ref:   
			view_imgs.append(self.normalize(re_ref_img[0]))
			re_size = re_size
			view_imgs = [F.interpolate(img.unsqueeze(0), size=re_size, mode='bilinear', align_corners=True)[0] for img in view_imgs]
			n_rows = 4+2

		else:
			re_size = re_size
			view_imgs = [F.interpolate(img.unsqueeze(0), size=re_size, mode='bilinear', align_corners=True)[0] for img in view_imgs]
			n_rows = 4+1

		if flow is not None:
			# print(flow.size())
			flow_f = flow_to_image(flow[0,0].cpu().unsqueeze(0).permute(0,2,3,1).contiguous().numpy())
			flow_f = F.interpolate(torch.tensor(flow_f).permute(0,3,1,2), size=re_size, mode='bilinear', align_corners=True)[0]
			flow_b = flow_to_image(flow[0,1].cpu().unsqueeze(0).permute(0,2,3,1).contiguous().numpy())
			flow_b = F.interpolate(torch.tensor(flow_b).permute(0,3,1,2), size=re_size, mode='bilinear', align_corners=True)[0]
			view_imgs += [flow_f, self.base_flow_map, flow_b]

		write_in_img = make_grid(view_imgs, nrow=n_rows)
		# else:
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

		coarse_l1_loss_record = 0
		coarse_ssim_loss_record = 0
		coarse_vgg_loss_record = 0
		coarse_loss_all_record = 0
		coarse_ce_loss_record = 0

		refine_l1_loss_record = 0
		refine_vgg_loss_record = 0
		refine_gdl_loss_record = 0
		refine_loss_all_record = 0

		refine_h_l1_loss_record = 0
		refine_h_vgg_loss_record = 0
		refine_h_gdl_loss_record = 0
		refine_h_loss_all_record = 0

		if self.args.high_res:
			refine_high_res_l1_loss_record = 0
			refine_high_res_vgg_loss_record = 0 # dont use vgg since network is too big
			refine_high_res_gdl_loss_record = 0
			refine_high_res_loss_all_record = 0
			loss_all_all_record = 0

		if self.args.re_ref:
			refine_re_ref_l1_loss_record = 0
			refine_re_ref_vgg_loss_record = 0 # dont use vgg since network is too big
			refine_re_ref_gdl_loss_record = 0
			refine_re_ref_loss_all_record = 0
			loss_all_all_record = 0

		loss_all_record = 0

		data_all_count = 0

		epoch_coarse_l1_loss_record = 0
		epoch_coarse_ssim_loss_record = 0
		epoch_coarse_vgg_loss_record = 0
		epoch_coarse_loss_all_record = 0
		epoch_coarse_ce_loss_record = 0

		epoch_refine_l1_loss_record = 0
		epoch_refine_vgg_loss_record = 0
		epoch_refine_gdl_loss_record = 0
		epoch_refine_loss_all_record = 0

		epoch_refine_h_l1_loss_record = 0
		epoch_refine_h_vgg_loss_record = 0
		epoch_refine_h_gdl_loss_record = 0
		epoch_refine_h_vgg_loss_record = 0
		epoch_refine_h_loss_all_record = 0

		if self.args.high_res:
			epoch_refine_high_res_l1_loss_record = 0
			epoch_refine_high_res_vgg_loss_record = 0
			epoch_refine_high_res_gdl_loss_record = 0
			epoch_refine_high_res_loss_all_record = 0
			epoch_loss_all_all_record = 0

		if self.args.re_ref:
			epoch_refine_re_ref_l1_loss_record = 0
			epoch_refine_re_ref_vgg_loss_record = 0
			epoch_refine_re_ref_gdl_loss_record = 0
			epoch_refine_re_ref_loss_all_record = 0
			epoch_loss_all_all_record = 0

		epoch_loss_all_record = 0
		

		epoch_data_all_count = 0
		# print("length", len(self.train_loader))
		for step, data in enumerate(self.train_loader):
			self.step = step
			load_time += time() - end
			end = time()
			# for tensorboard
			self.global_step += 1
			# forward pass
			# if not self.args.high_res:
			#   x, fg_mask, gt = self.get_input(data)
			#   x = x.cuda(self.args.rank, non_blocking=True)
			#   fg_mask = fg_mask.cuda(self.args.rank, non_blocking=True)
			#   gt = gt.cuda(self.args.rank, non_blocking=True)
			# else:
			x, seg, fg_mask, gt_x, gt_seg = self.get_input(data)
			x = x.cuda(self.args.rank, non_blocking=True)
			seg = seg.cuda(self.args.rank, non_blocking=True)  if self.args.mode == 'xs2xs' else None
			fg_mask = fg_mask.cuda(self.args.rank, non_blocking=True) if self.args.mode == 'xs2xs' else None
			gt_seg = gt_seg.cuda(self.args.rank, non_blocking=True) if self.args.mode == 'xs2xs' else None
			gt_x = gt_x.cuda(self.args.rank, non_blocking=True)
			

			batch_size_p = x.size(0)
			data_all_count += batch_size_p
			epoch_data_all_count += batch_size_p

			# if not self.args.high_res:
			#   coarse_img, refined_imgs, _, seg = self.model(x, fg_mask, gt=gt)
			#   if not self.args.lock_coarse and not self.args.lock_retrain:
			#       loss_dict = self.coarse_RGBLoss(coarse_img, gt[:, :3], False)
			#       if self.args.mode == 'xs2xs':
			#          loss_dict['ce_loss'] = self.args.ce_weight*self.SegLoss(seg, torch.argmax(gt[:,3:], dim=1))   
			#   else:
			#       loss_dict = OrderedDict()
			
			#   for i in range(self.args.n_scales):
			#       # print(i, refined_imgs[-i].size())
			#       loss_dict.update(self.refine_RGBLoss(refined_imgs[-i-1], F.interpolate(gt[:,:3], scale_factor=(1/2)**i, mode='bilinear', align_corners=True),\
			#                                                            refine_scale=1/(2**i), step=self.global_step, normed=False))               
			# else:x
			coarse_img, refined_imgs, refined_h_img, re_refd_img, seg, attn_flow = self.model(x, fg_mask, seg=seg, gt_x=gt_x, gt_seg=gt_seg)
			scaled_gt_x = F.interpolate(gt_x, scale_factor=0.5, mode='bilinear', align_corners=True) if self.args.high_res else gt_x
			if not self.args.lock_coarse and not self.args.lock_retrain:
				loss_dict = self.coarse_RGBLoss(coarse_img, scaled_gt_x, False)
				if self.args.mode == 'xs2xs':
				   loss_dict['ce_loss'] = self.args.ce_weight*self.SegLoss(seg, torch.argmax(gt_seg, dim=1))   
			else:
				loss_dict = OrderedDict()

			for i in range(self.args.n_scales):
				# print(i, refined_imgs[-i].size())
				loss_dict.update(self.refine_RGBLoss(refined_imgs[-i-1], F.interpolate(scaled_gt_x, scale_factor=(1/2)**i, mode='bilinear', align_corners=True),\
																	 refine_scale=1/(2**i), step=self.global_step, normed=False))   
			if self.args.high_res:
				loss_dict.update(self.refine_RGBLoss(refined_h_img, gt_x, refine_scale=1, step=self.global_step, normed=False, res=2))

			if self.args.re_ref:
				loss_dict.update(self.refine_RGBLoss(re_refd_img, gt_x, refine_scale=1, step=self.global_step, normed=False, res=3))

			# loss and accuracy
			loss = 0
			for i in loss_dict.values():
				loss += torch.mean(i)
			loss_dict['loss_all'] = loss
			
			self.sync(loss_dict)

			# backward pass
			self.coarse_optimizer.zero_grad()
			self.refine_optimizer.zero_grad()
			if self.args.high_res:
				self.high_res_optimizer.zero_grad()
			if self.args.re_ref:
				self.re_ref_optimizer.zero_grad()

			loss_dict['loss_all'].backward()
			# init step in the first train steps
			if not self.args.lock_coarse and not self.args.lock_low:
				self.coarse_optimizer.step()
			if not self.args.lock_low:
				self.refine_optimizer.step()
			if self.args.high_res:
				self.high_res_optimizer.step()
			if self.args.re_ref:
				self.re_ref_optimizer.step()
			comp_time += time() - end
			end = time()

			# update coarse record 
			if not self.args.lock_coarse:
				coarse_l1_loss_record   += batch_size_p*loss_dict['l1_loss'].item()
				coarse_ssim_loss_record += batch_size_p*loss_dict['ssim_loss'].item()
				coarse_vgg_loss_record  += batch_size_p*loss_dict['vgg_loss'].item()
				coarse_loss_all_record  = (coarse_l1_loss_record + coarse_ssim_loss_record + coarse_vgg_loss_record)
				coarse_ce_loss_record   += batch_size_p*loss_dict['ce_loss'].item() if self.args.mode=='xs2xs' else 0

			# if not self.args.lock_low:
			# update refine 1.0 record 
			refine_l1_loss_record   += batch_size_p*loss_dict['refine_1.00_l1_loss'].item()
			refine_vgg_loss_record  += batch_size_p*loss_dict['refine_1.00_vgg_loss'].item()
			refine_gdl_loss_record  += batch_size_p*loss_dict['refine_1.00_gdl_loss'].item()
			refine_loss_all_record  = (refine_l1_loss_record + refine_vgg_loss_record + refine_gdl_loss_record)


			if self.args.n_scales == 2:
				# update refine 0.5 record 
				refine_h_l1_loss_record   += batch_size_p*loss_dict['refine_0.50_l1_loss'].item()
				refine_h_vgg_loss_record  += batch_size_p*loss_dict['refine_0.50_vgg_loss'].item()
				refine_h_gdl_loss_record  += batch_size_p*loss_dict['refine_0.50_gdl_loss'].item()
				refine_h_loss_all_record  = (refine_h_l1_loss_record + refine_h_vgg_loss_record + refine_h_gdl_loss_record)

			# update all loss record 
			loss_all_record = coarse_loss_all_record + coarse_ce_loss_record + refine_loss_all_record + refine_h_loss_all_record

			if self.args.high_res:
				# update high res record
				refine_high_res_l1_loss_record   += batch_size_p*loss_dict['refine_2_l1_loss'].item()
				refine_high_res_vgg_loss_record  += batch_size_p*loss_dict['refine_2_vgg_loss'].item()
				refine_high_res_gdl_loss_record  += batch_size_p*loss_dict['refine_2_gdl_loss'].item()
				refine_high_res_loss_all_record  = (refine_high_res_l1_loss_record + refine_high_res_vgg_loss_record + refine_high_res_gdl_loss_record)

				loss_all_all_record = loss_all_record + refine_high_res_loss_all_record

			if self.args.re_ref:
				# update high res record
				refine_re_ref_l1_loss_record   += batch_size_p*loss_dict['refine_3_l1_loss'].item()
				refine_re_ref_vgg_loss_record  += batch_size_p*loss_dict['refine_3_vgg_loss'].item()
				refine_re_ref_gdl_loss_record  += batch_size_p*loss_dict['refine_3_gdl_loss'].item()
				refine_re_ref_loss_all_record  = (refine_re_ref_l1_loss_record + refine_re_ref_vgg_loss_record + refine_re_ref_gdl_loss_record)

				loss_all_all_record = loss_all_record + refine_re_ref_loss_all_record

			# updata epoch record
			if not self.args.lock_coarse:
				epoch_coarse_l1_loss_record   += batch_size_p*loss_dict['l1_loss'].item()
				epoch_coarse_ssim_loss_record += batch_size_p*loss_dict['ssim_loss'].item()
				epoch_coarse_vgg_loss_record  += batch_size_p*loss_dict['vgg_loss'].item()
				epoch_coarse_ce_loss_record   += batch_size_p*loss_dict['ce_loss'].item() if self.args.mode=='xs2xs' else 0
			# if not self.args.lock_low:
			epoch_refine_l1_loss_record   += batch_size_p*loss_dict['refine_1.00_l1_loss'].item()
			epoch_refine_vgg_loss_record  += batch_size_p*loss_dict['refine_1.00_vgg_loss'].item()
			epoch_refine_gdl_loss_record  += batch_size_p*loss_dict['refine_1.00_gdl_loss'].item()
			if self.args.n_scales == 2:
				epoch_refine_h_l1_loss_record   += batch_size_p*loss_dict['refine_0.50_l1_loss'].item()
				epoch_refine_h_vgg_loss_record  += batch_size_p*loss_dict['refine_0.50_vgg_loss'].item()
				epoch_refine_h_gdl_loss_record  += batch_size_p*loss_dict['refine_0.50_gdl_loss'].item()
			if self.args.high_res:
				epoch_refine_high_res_l1_loss_record   += batch_size_p*loss_dict['refine_2_l1_loss'].item()
				epoch_refine_high_res_vgg_loss_record  += batch_size_p*loss_dict['refine_2_vgg_loss'].item()
				epoch_refine_high_res_gdl_loss_record  += batch_size_p*loss_dict['refine_2_gdl_loss'].item()
			if self.args.re_ref:
				epoch_refine_re_ref_l1_loss_record   += batch_size_p*loss_dict['refine_3_l1_loss'].item()
				epoch_refine_re_ref_vgg_loss_record  += batch_size_p*loss_dict['refine_3_vgg_loss'].item()
				epoch_refine_re_ref_gdl_loss_record  += batch_size_p*loss_dict['refine_3_gdl_loss'].item()

			if self.args.rank == 0:
				# add info to tensorboard
				info = {key:value.item() for key,value in loss_dict.items()}
				self.writer.add_scalars("losses", info, self.global_step)
				# print
				if self.step % self.args.disp_interval == 0:
					if data_all_count != 0:
						coarse_l1_loss_record   /= data_all_count
						coarse_ssim_loss_record /= data_all_count
						coarse_vgg_loss_record  /= data_all_count
						coarse_loss_all_record  /= data_all_count
						coarse_ce_loss_record   /= data_all_count

						refine_l1_loss_record   /= data_all_count
						refine_vgg_loss_record  /= data_all_count
						refine_gdl_loss_record  /= data_all_count
						refine_loss_all_record  /= data_all_count

						refine_h_l1_loss_record   /= data_all_count
						refine_h_vgg_loss_record  /= data_all_count
						refine_h_gdl_loss_record  /= data_all_count
						refine_h_loss_all_record  /= data_all_count

						loss_all_record /= data_all_count
						if self.args.high_res:
							refine_high_res_l1_loss_record   /= data_all_count
							refine_high_res_vgg_loss_record  /= data_all_count
							refine_high_res_gdl_loss_record  /= data_all_count
							refine_high_res_loss_all_record  /= data_all_count
							loss_all_all_record /= data_all_count
						if self.args.re_ref:
							refine_re_ref_l1_loss_record   /= data_all_count
							refine_re_ref_vgg_loss_record  /= data_all_count
							refine_re_ref_gdl_loss_record  /= data_all_count
							refine_re_ref_loss_all_record  /= data_all_count
							loss_all_all_record /= data_all_count

					if self.args.high_res:
						self.args.logger.info(
							'Epoch [{epoch:d}/{tot_epoch:d}][{cur_batch:d}/{tot_batch:d}] '
							'load [{load_time:.3f}s] comp [{comp_time:.3f}s] '
							'\n\t\tcoarse l1 [{l1:.4f}] vgg [{vgg:.4f}] ssim [{ssim:.4f}] rgb_all [{rgb_all:.4f}] ce  [{ce:.4f}] '
							'\n\t\tref0.5 l1 [{refine_h_l1:.4f}] vgg [{refine_h_vgg:.4f}] gdl  [{refine_h_gdl:.4f}] rgb_all [{refine_h_all:.4f}]'
							'\n\t\trefine l1 [{refine_l1:.4f}] vgg [{refine_vgg:.4f}] gdl  [{refine_gdl:.4f}] rgb_all [{refine_all:.4f}] low all [{low_all:.4f}]'
							'\n\t\thighre l1 [{high_res_l1:.4f}] gdl  [{high_res_gdl:.4f}] rgb_all [{high_res_all:.4f}] all [{all:.4f}]'.format(
								epoch=self.epoch, tot_epoch=self.args.epochs,
								cur_batch=self.step+1, tot_batch=len(self.train_loader),
								load_time=load_time, comp_time=comp_time,
								l1=coarse_l1_loss_record, vgg=coarse_vgg_loss_record, ssim=coarse_ssim_loss_record, rgb_all=coarse_loss_all_record, ce=coarse_ce_loss_record,
								refine_h_l1=refine_h_l1_loss_record, refine_h_vgg=refine_h_vgg_loss_record, refine_h_gdl=refine_h_gdl_loss_record, refine_h_all=refine_h_loss_all_record,
								refine_l1=refine_l1_loss_record, refine_vgg=refine_vgg_loss_record, refine_gdl=refine_gdl_loss_record, refine_all=refine_loss_all_record, low_all=loss_all_record,
								high_res_l1=refine_high_res_l1_loss_record, high_res_gdl=refine_high_res_gdl_loss_record, high_res_all=refine_high_res_loss_all_record, all=loss_all_all_record
							)
						)
					elif self.args.re_ref:
						self.args.logger.info(
							'Epoch [{epoch:d}/{tot_epoch:d}][{cur_batch:d}/{tot_batch:d}] '
							'load [{load_time:.3f}s] comp [{comp_time:.3f}s] '
							'\n\t\tcoarse l1 [{l1:.4f}] vgg [{vgg:.4f}] ssim [{ssim:.4f}] rgb_all [{rgb_all:.4f}] ce  [{ce:.4f}] '
							'\n\t\tref0.5 l1 [{refine_h_l1:.4f}] vgg [{refine_h_vgg:.4f}] gdl  [{refine_h_gdl:.4f}] rgb_all [{refine_h_all:.4f}]'
							'\n\t\trefine l1 [{refine_l1:.4f}] vgg [{refine_vgg:.4f}] gdl  [{refine_gdl:.4f}] rgb_all [{refine_all:.4f}] low all [{low_all:.4f}]'
							'\n\t\tre  re l1 [{re_ref_l1:.4f}] gdl  [{re_ref_gdl:.4f}] rgb_all [{re_ref_all:.4f}] all [{all:.4f}]'.format(
								epoch=self.epoch, tot_epoch=self.args.epochs,
								cur_batch=self.step+1, tot_batch=len(self.train_loader),
								load_time=load_time, comp_time=comp_time,
								l1=coarse_l1_loss_record, vgg=coarse_vgg_loss_record, ssim=coarse_ssim_loss_record, rgb_all=coarse_loss_all_record, ce=coarse_ce_loss_record,
								refine_h_l1=refine_h_l1_loss_record, refine_h_vgg=refine_h_vgg_loss_record, refine_h_gdl=refine_h_gdl_loss_record, refine_h_all=refine_h_loss_all_record,
								refine_l1=refine_l1_loss_record, refine_vgg=refine_vgg_loss_record, refine_gdl=refine_gdl_loss_record, refine_all=refine_loss_all_record, low_all=loss_all_record,
								re_ref_l1=refine_re_ref_l1_loss_record, re_ref_gdl=refine_re_ref_gdl_loss_record, re_ref_all=refine_re_ref_loss_all_record, all=loss_all_all_record
							)
						)					
					else:
						self.args.logger.info(
							'Epoch [{epoch:d}/{tot_epoch:d}][{cur_batch:d}/{tot_batch:d}] '
							'load [{load_time:.3f}s] comp [{comp_time:.3f}s] '
							'\n\t\tcoarse l1 [{l1:.4f}] vgg [{vgg:.4f}] ssim [{ssim:.4f}] rgb_all [{rgb_all:.4f}] ce  [{ce:.4f}] '
							'\n\t\tref0.5 l1 [{refine_h_l1:.4f}] vgg [{refine_h_vgg:.4f}] gdl  [{refine_h_gdl:.4f}] rgb_all [{refine_h_all:.4f}]'
							'\n\t\trefine l1 [{refine_l1:.4f}] vgg [{refine_vgg:.4f}] gdl  [{refine_gdl:.4f}] rgb_all [{refine_all:.4f}] all [{all:.4f}]'.format(
								epoch=self.epoch, tot_epoch=self.args.epochs,
								cur_batch=self.step+1, tot_batch=len(self.train_loader),
								load_time=load_time, comp_time=comp_time,
								l1=coarse_l1_loss_record, vgg=coarse_vgg_loss_record, ssim=coarse_ssim_loss_record, rgb_all=coarse_loss_all_record, ce=coarse_ce_loss_record,
								refine_h_l1=refine_h_l1_loss_record, refine_h_vgg=refine_h_vgg_loss_record, refine_h_gdl=refine_h_gdl_loss_record, refine_h_all=refine_h_loss_all_record,
								refine_l1=refine_l1_loss_record, refine_vgg=refine_vgg_loss_record, refine_gdl=refine_gdl_loss_record, refine_all=refine_loss_all_record, all=loss_all_record
							)
						)
					comp_time = 0
					load_time = 0

					coarse_l1_loss_record = 0
					coarse_ssim_loss_record = 0
					coarse_vgg_loss_record = 0
					coarse_loss_all_record = 0
					coarse_ce_loss_record = 0

					refine_l1_loss_record = 0
					refine_vgg_loss_record = 0
					refine_gdl_loss_record = 0
					refine_loss_all_record = 0

					refine_h_l1_loss_record = 0
					refine_h_vgg_loss_record = 0
					refine_h_gdl_loss_record = 0
					refine_h_loss_all_record = 0

					if self.args.high_res:
						refine_high_res_l1_loss_record = 0
						refine_high_res_vgg_loss_record = 0
						refine_high_res_gdl_loss_record = 0
						refine_high_res_loss_all_record = 0
						loss_all_all_record = 0

					if self.args.re_ref:
						refine_re_ref_l1_loss_record = 0
						refine_re_ref_vgg_loss_record = 0
						refine_re_ref_gdl_loss_record = 0
						refine_re_ref_loss_all_record = 0
						loss_all_all_record = 0

					loss_all_record = 0
					
					data_all_count = 0

				if self.step % 30 == 0: 
					if self.args.high_res:
						image_set = self.prepare_image_set(data, coarse_img.cpu(), [ refined_img.cpu() for refined_img in refined_imgs], seg.cpu() if self.args.mode=='xs2xs' else None, high_res_img=refined_h_img.cpu(), flow=attn_flow)
					elif self.args.re_ref:
						image_set = self.prepare_image_set(data, coarse_img.cpu(), [ refined_img.cpu() for refined_img in refined_imgs], seg.cpu() if self.args.mode=='xs2xs' else None, re_ref_img=re_refd_img.cpu(), flow=attn_flow)
					else:
						image_set = self.prepare_image_set(data, coarse_img.cpu(), [ refined_img.cpu() for refined_img in refined_imgs], seg.cpu() if self.args.mode=='xs2xs' else None, flow=attn_flow)
					self.writer.add_image('image_{}'.format(self.global_step), image_set, self.global_step)
		

		# epoch summary
		epoch_coarse_loss_all_record  = (epoch_coarse_l1_loss_record + epoch_coarse_ssim_loss_record + epoch_coarse_vgg_loss_record)
		epoch_refine_loss_all_record  = (epoch_refine_l1_loss_record + epoch_refine_vgg_loss_record + epoch_refine_gdl_loss_record)
		epoch_refine_h_loss_all_record  = (epoch_refine_h_l1_loss_record + epoch_refine_h_vgg_loss_record + epoch_refine_h_gdl_loss_record)
		epoch_loss_all_record = epoch_coarse_loss_all_record + epoch_coarse_ce_loss_record + epoch_refine_loss_all_record + epoch_refine_h_loss_all_record

		if self.args.high_res:
			epoch_refine_high_res_loss_all_record  = (epoch_refine_high_res_l1_loss_record+ epoch_refine_high_res_vgg_loss_record + epoch_refine_high_res_gdl_loss_record)
			epoch_loss_all_all_record = epoch_loss_all_record + epoch_refine_high_res_loss_all_record

		if self.args.re_ref:
			epoch_refine_re_ref_loss_all_record  = (epoch_refine_re_ref_l1_loss_record+ epoch_refine_re_ref_vgg_loss_record + epoch_refine_re_ref_gdl_loss_record)
			epoch_loss_all_all_record = epoch_loss_all_record + epoch_refine_high_res_loss_all_record

		epoch_coarse_l1_loss_record   /= epoch_data_all_count
		epoch_coarse_ssim_loss_record /= epoch_data_all_count
		epoch_coarse_vgg_loss_record  /= epoch_data_all_count
		epoch_coarse_loss_all_record  /= epoch_data_all_count
		epoch_coarse_ce_loss_record   /= epoch_data_all_count

		epoch_refine_l1_loss_record   /= epoch_data_all_count
		epoch_refine_vgg_loss_record  /= epoch_data_all_count
		epoch_refine_gdl_loss_record  /= epoch_data_all_count
		epoch_refine_loss_all_record  /= epoch_data_all_count

		epoch_refine_h_l1_loss_record   /= epoch_data_all_count
		epoch_refine_h_vgg_loss_record  /= epoch_data_all_count
		epoch_refine_h_gdl_loss_record  /= epoch_data_all_count
		epoch_refine_h_loss_all_record  /= epoch_data_all_count

		if self.args.high_res:
			epoch_refine_high_res_l1_loss_record   /= epoch_data_all_count
			epoch_refine_high_res_vgg_loss_record  /= epoch_data_all_count
			epoch_refine_high_res_gdl_loss_record  /= epoch_data_all_count
			epoch_refine_high_res_loss_all_record  /= epoch_data_all_count
			epoch_loss_all_all_record /= epoch_data_all_count

		if self.args.re_ref:
			epoch_refine_re_ref_l1_loss_record   /= epoch_data_all_count
			epoch_refine_re_ref_vgg_loss_record  /= epoch_data_all_count
			epoch_refine_re_ref_gdl_loss_record  /= epoch_data_all_count
			epoch_refine_re_ref_loss_all_record  /= epoch_data_all_count
			epoch_loss_all_all_record /= epoch_data_all_count

		epoch_loss_all_record /= epoch_data_all_count

		if self.args.rank == 0:
			if self.args.high_res:
				self.args.logger.info(
					'Epoch [{epoch:d}/{tot_epoch:d}] '
					'\n\t\t\tcoarse l1 [{l1:.4f}] vgg [{vgg:.4f}] ssim [{ssim:.4f}] rgb_all [{rgb_all:.4f}] ce  [{ce:.4f}] '
					'\n\t\t\tref0.5 l1 [{refine_h_l1:.4f}] vgg [{refine_h_vgg:.4f}] gdl  [{refine_h_gdl:.4f}] rgb_all [{refine_h_all:.4f}]'
					'\n\t\t\trefine l1 [{refine_l1:.4f}] vgg [{refine_vgg:.4f}] gdl  [{refine_gdl:.4f}] rgb_all [{refine_all:.4f}] low all [{low_all:.4f}]'
					'\n\t\t\thighre l1 [{high_res_l1:.4f}] vgg [{high_res_vgg:.4f}] gdl  [{high_res_gdl:.4f}] rgb_all [{high_res_all:.4f}] all [{all:.4f}]'.format(
						epoch=self.epoch, tot_epoch=self.args.epochs,
						l1=epoch_coarse_l1_loss_record, vgg=epoch_coarse_vgg_loss_record, ssim=epoch_coarse_ssim_loss_record, rgb_all=epoch_coarse_loss_all_record, ce=epoch_coarse_ce_loss_record,
						refine_h_l1=epoch_refine_h_l1_loss_record, refine_h_vgg=epoch_refine_h_vgg_loss_record, refine_h_gdl=epoch_refine_h_gdl_loss_record, refine_h_all=epoch_refine_h_loss_all_record,
						refine_l1=epoch_refine_l1_loss_record, refine_vgg=epoch_refine_vgg_loss_record, refine_gdl=epoch_refine_gdl_loss_record, refine_all=epoch_refine_loss_all_record, low_all=epoch_loss_all_record,
						high_res_l1=epoch_refine_high_res_l1_loss_record, high_res_vgg=epoch_refine_high_res_vgg_loss_record, high_res_gdl=epoch_refine_high_res_gdl_loss_record, high_res_all=epoch_refine_high_res_loss_all_record, all=epoch_loss_all_all_record
					)
				)
			elif self.args.re_ref:
				self.args.logger.info(
					'Epoch [{epoch:d}/{tot_epoch:d}] '
					'\n\t\t\tcoarse l1 [{l1:.4f}] vgg [{vgg:.4f}] ssim [{ssim:.4f}] rgb_all [{rgb_all:.4f}] ce  [{ce:.4f}] '
					'\n\t\t\tref0.5 l1 [{refine_h_l1:.4f}] vgg [{refine_h_vgg:.4f}] gdl  [{refine_h_gdl:.4f}] rgb_all [{refine_h_all:.4f}]'
					'\n\t\t\trefine l1 [{refine_l1:.4f}] vgg [{refine_vgg:.4f}] gdl  [{refine_gdl:.4f}] rgb_all [{refine_all:.4f}] low all [{low_all:.4f}]'
					'\n\t\t\tre  re l1 [{re_ref_l1:.4f}] vgg [{re_ref_vgg:.4f}] gdl  [{re_ref_gdl:.4f}] rgb_all [{re_ref_all:.4f}] all [{all:.4f}]'.format(
						epoch=self.epoch, tot_epoch=self.args.epochs,
						l1=epoch_coarse_l1_loss_record, vgg=epoch_coarse_vgg_loss_record, ssim=epoch_coarse_ssim_loss_record, rgb_all=epoch_coarse_loss_all_record, ce=epoch_coarse_ce_loss_record,
						refine_h_l1=epoch_refine_h_l1_loss_record, refine_h_vgg=epoch_refine_h_vgg_loss_record, refine_h_gdl=epoch_refine_h_gdl_loss_record, refine_h_all=epoch_refine_h_loss_all_record,
						refine_l1=epoch_refine_l1_loss_record, refine_vgg=epoch_refine_vgg_loss_record, refine_gdl=epoch_refine_gdl_loss_record, refine_all=epoch_refine_loss_all_record, low_all=epoch_loss_all_record,
						re_ref_l1=epoch_refine_re_ref_l1_loss_record, re_ref_vgg=epoch_refine_re_ref_vgg_loss_record, re_ref_gdl=epoch_refine_re_ref_gdl_loss_record, re_ref_all=epoch_refine_re_ref_loss_all_record, all=epoch_loss_all_all_record
					)
				)
			else:
				self.args.logger.info(
					'Epoch [{epoch:d}/{tot_epoch:d}] '
					'\n\t\t\tcoarse l1 [{l1:.4f}] vgg [{vgg:.4f}] ssim [{ssim:.4f}] rgb_all [{rgb_all:.4f}] ce  [{ce:.4f}] '
					'\n\t\t\tref0.5 l1 [{refine_h_l1:.4f}] vgg [{refine_h_vgg:.4f}] gdl  [{refine_h_gdl:.4f}] rgb_all [{refine_h_all:.4f}]'
					'\n\t\t\trefine l1 [{refine_l1:.4f}] vgg [{refine_vgg:.4f}] gdl  [{refine_gdl:.4f}] rgb_all [{refine_all:.4f}] all [{all:.4f}]'.format(
						epoch=self.epoch,tot_epoch=self.args.epochs,
						l1=epoch_coarse_l1_loss_record, vgg=epoch_coarse_vgg_loss_record, ssim=epoch_coarse_ssim_loss_record, rgb_all=epoch_coarse_loss_all_record, ce=epoch_coarse_ce_loss_record,
						refine_h_l1=epoch_refine_h_l1_loss_record, refine_h_vgg=epoch_refine_h_vgg_loss_record, refine_h_gdl=epoch_refine_h_gdl_loss_record, refine_h_all=epoch_refine_h_loss_all_record,
						refine_l1=epoch_refine_l1_loss_record, refine_vgg=epoch_refine_vgg_loss_record, refine_gdl=epoch_refine_gdl_loss_record, refine_all=epoch_refine_loss_all_record, all=epoch_loss_all_record
					)
				)

	def validate(self, val_coarse=True):
		self.args.logger.info('Validation epoch {} started'.format(self.epoch))
		self.model.eval()

		val_criteria = {
			# 'l1': AverageMeter(),
			# 'psnr':AverageMeter(),
			# 'ssim':AverageMeter(),
			# 'iou':AverageMeter(),
			# 'vgg':AverageMeter()
			'refine_l1': AverageMeter(),
			'refine_psnr':AverageMeter(),
			'refine_ssim':AverageMeter(),
			'refine_vgg':AverageMeter()
		}
		if val_coarse:
			val_criteria['l1'] = AverageMeter()
			val_criteria['psnr'] = AverageMeter()
			val_criteria['ssim'] = AverageMeter()
			if self.args.mode == 'xs2xs':
				val_criteria['iou'] = AverageMeter()
			val_criteria['vgg'] = AverageMeter()
		if self.args.high_res:
			val_criteria['h_l1']  = AverageMeter()
			val_criteria['h_psnr'] = AverageMeter()
			val_criteria['h_ssim'] = AverageMeter()
			val_criteria['h_vgg']  = AverageMeter()
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
				# if not self.args.high_res:
				#   x, fg_mask, gt = self.get_input(data)
				#   size = x.size(0)
				#   x = x.cuda(self.args.rank, non_blocking=True)
				#   fg_mask = fg_mask.cuda(self.args.rank, non_blocking=True)
				#   gt = gt.cuda(self.args.rank, non_blocking=True)
					
				#   coarse_img, refine_imgs, seg = self.model(x, fg_mask)
				#   coarse_gt = gt[:,:3]
				#   gt_seg = gt[:, 3:]
				# else:
				x, seg, fg_mask, gt_x, gt_seg = self.get_input(data)
				size = x.size(0)
				x = x.cuda(self.args.rank, non_blocking=True)
				seg = seg.cuda(self.args.rank, non_blocking=True)  if self.args.mode == 'xs2xs' else None
				fg_mask = fg_mask.cuda(self.args.rank, non_blocking=True)  if self.args.mode == 'xs2xs' else None
				gt_x = gt_x.cuda(self.args.rank, non_blocking=True)
				gt_seg = gt_seg.cuda(self.args.rank, non_blocking=True)  if self.args.mode == 'xs2xs' else None

				coarse_img, refine_imgs, refined_h_img, seg, attn_flow = self.model(x, fg_mask, seg=seg, gt_x=gt_x, gt_seg=gt_seg)
				coarse_gt = F.interpolate(gt_x, scale_factor=0.5, mode='bilinear', align_corners=True) if self.args.high_res else gt_x

				if self.args.high_res:
					step_losses['h_l1'] =   self.L1Loss(refined_h_img, gt_x)
					step_losses['h_psnr'] = self.PSNRLoss((refined_h_img+1)/2, (gt_x+1)/2)
					step_losses['h_ssim'] = 1-self.SSIMLoss(refined_h_img, gt_x)
					step_losses['h_vgg'] =  self.VGGCosLoss(refined_h_img, gt_x, False)

				# rgb criteria
				if val_coarse:
					step_losses['l1'] =   self.L1Loss(coarse_img, coarse_gt)
					step_losses['psnr'] = self.PSNRLoss((coarse_img+1)/2, (coarse_gt+1)/2)
					step_losses['ssim'] = 1-self.SSIMLoss(coarse_img, coarse_gt)
					if self.args.mode == 'xs2xs':
						step_losses['iou'] =  self.IoULoss(torch.argmax(seg, dim=1), torch.argmax(gt_seg, dim=1))
					step_losses['vgg'] =  self.VGGCosLoss(coarse_img, coarse_gt, False)

				step_losses['refine_l1'] =   self.L1Loss(refine_imgs[-1], coarse_gt)
				step_losses['refine_psnr'] = self.PSNRLoss((refine_imgs[-1]+1)/2, (coarse_gt+1)/2)
				step_losses['refine_ssim'] = 1-self.SSIMLoss(refine_imgs[-1], coarse_gt)
				step_losses['refine_vgg'] =  self.VGGCosLoss(refine_imgs[-1], coarse_gt, False)

				self.sync(step_losses) # sum
				for key in list(val_criteria.keys()):
					val_criteria[key].update(step_losses[key].cpu().item(), size*self.args.gpus)

				# if self.args.syn_type == 'extra':
				# 	imgs = []
				# 	segs = []
				# 	img = img[0].unsqueeze(0)
				# 	seg = seg[0].unsqueeze(0)
				# 	x = x[0].unsqueeze(0)
				# 	for i in range(self.args.extra_length):
				# 		if i!=0:
				# 			x = torch.cat([x[:,3:6], img, x[:, 26:46], seg_fil], dim=1).cuda(self.args.rank, non_blocking=True)
				# 			img, seg = self.model(x)
				# 		seg_fil = torch.argmax(seg, dim=1)
				# 		seg_fil = transform_seg_one_hot(seg_fil, 20, cuda=True)*2-1
				# 		imgs.append(img)
				# 		segs.append(seg_fil)
						
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
						if not self.args.high_res:
							image_set = self.prepare_image_set(data, coarse_img.cpu(), [ refined_img.cpu() for refined_img in refine_imgs], seg.cpu() if self.args.mode=='xs2xs' else None, flow=attn_flow)
						else:
							image_set = self.prepare_image_set(data, coarse_img.cpu(), [ refined_img.cpu() for refined_img in refine_imgs], seg.cpu() if self.args.mode=='xs2xs' else None, high_res_img=refined_h_img.cpu(), flow=attn_flow)
						image_name = 'e{}_img_{}'.format(self.epoch, self.step)
						self.writer.add_image(image_name, image_set, self.step)

		if self.args.rank == 0:

			logs = "refine L1: {r_l1:.4f}\tPSNR: {r_psnr:.4f}\tSSIM: {r_ssim:.4f}\tvgg: {r_vgg:.4f}".format(
						r_l1=val_criteria['refine_l1'].avg,
						r_psnr=val_criteria['refine_psnr'].avg,
						r_ssim=val_criteria['refine_ssim'].avg,
						r_vgg = val_criteria['refine_vgg'].avg
				) + "\n"

			if val_coarse:
				coarse_logs = "coarse L1: {l1:.4f}\tPSNR: {psnr:.4f}\tSSIM: {ssim:.4f}\tvgg: {vgg:.4f}".format(
						l1=val_criteria['l1'].avg,
						psnr=val_criteria['psnr'].avg,
						ssim=val_criteria['ssim'].avg,
						vgg = val_criteria['vgg'].avg
				)
				logs = coarse_logs + "\n" + logs
			if self.args.high_res:
				highres_logs = "highre L1: {h_l1:.4f}\tPSNR: {h_psnr:.4f}\tSSIM: {h_ssim:.4f}\tvgg: {h_vgg:.4f}".format(
						h_l1=val_criteria['h_l1'].avg,
						h_psnr=val_criteria['h_psnr'].avg,
						h_ssim=val_criteria['h_ssim'].avg,
						h_vgg = val_criteria['h_vgg'].avg
				)
				logs = logs + highres_logs + '\n'

			logs = "Epoch [{epoch:d}]\n".format(epoch=self.epoch) + logs
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

	def cycgen(self):
		assert self.args.rank == 0 # only allow single worker
		if self.args.syn_type == 'extra':
			assert self.args.interval == 1
			load_dir_split = 'gt'
			save_dir_split = 'extra_x{}'.format(self.args.vid_length)
		elif self.args.syn_type == 'inter':
			save_dir_split = 'inter_x{:d}'.format(int(2/self.args.interval))
			load_dir_split = 'inter_x{:d}'.format(int(1/self.args.interval)) if self.args.interval<1 else 'gt'
		load_img_dir = 'cycgen/cityscape/{}x{}/{}'.format(self.args.input_h, self.args.input_w, load_dir_split)
		save_img_dir = 'cycgen/cityscape/{}x{}/{}'.format(self.args.input_h, self.args.input_w, save_dir_split)

		with open('/data/linz/proj/Dataset/Cityscape/load_files/root_clip.pkl', 'rb') as f:
			clips = pickle.load(f)
			clips_dir = clips['val'][:61] # onlye generate 0-60
		load_clip_dirs = [load_img_dir+'/'+clip_dir[0] for clip_dir in clips_dir]
		end = time()
		for clip_ind, load_clip_dir in enumerate(load_clip_dirs):
			load_img_files = glob.glob(load_clip_dir+"/*.png")
			load_img_files.sort()
			load_imgs = rgb_load(load_img_files)
			for i in range(len(load_imgs)):
				load_imgs[i] = transforms.functional.normalize( 
										transforms.functional.to_tensor(
											load_imgs[i]
										),  (0.5, 0.5, 0.5),(0.5, 0.5, 0.5)
									).unsqueeze(0)
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
				input_imgs = load_imgs[:2]
				input_imgs = [t.cuda(self.args.rank) for t in input_imgs]
				img1_ind = float(load_img_files[0].split('/')[-1][:-4])
				img2_ind = float(load_img_files[1].split('/')[-1][:-4])
				time_interval = img2_ind - img1_ind
				pred_img_name_prefix = '/'.join(load_img_files[0].split('/')[-4:-1])
				save_img_prefix = save_img_dir + '/' + pred_img_name_prefix
				if not os.path.exists(save_img_prefix):
					os.makedirs(save_img_prefix)			

				for step in range(self.args.vid_length):
					pred_img_ind = img2_ind + (step+1)*time_interval
					# following 5 lines make integer part _ _ 
					pred_img_ind = str(pred_img_ind)
					pred_img_ind_split = pred_img_ind.split('.')
					pred_img_ind_int = '{:0>2d}'.format(int(pred_img_ind_split[0]))
					pred_img_ind_split[0] = pred_img_ind_int
					pred_img_ind = '.'.join(pred_img_ind_split)					

					pred_img_name = save_img_prefix + '/' + pred_img_ind+".png"

					x = torch.cat(input_imgs, dim=1)
					x = x.cuda(self.args.rank, non_blocking=True)
					seg = None
					fg_mask = None
					gt_x = None
					gt_seg =  None

					coarse_img, refine_imgs, refined_h_img, seg, attn_flow = self.model(x, fg_mask, seg=seg, gt_x=gt_x, gt_seg=gt_seg)
					pred_img = refine_imgs[-1]
					input_imgs = [input_imgs[1]] + [pred_img]
					pred_img = pred_img[0]
					pred_img = self.normalize(pred_img)
					save_image(pred_img, pred_img_name)
				# save gt
				for step in range(2):
					load_img_name = '/'.join(load_img_files[step].split('/')[-4:])
					save_img_name = save_img_dir + '/' + load_img_name
					shutil.copyfile(load_img_files[step], save_img_name)

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
			elif self.args.pretrained_low :#and not self.args.resume:
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

