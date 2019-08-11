import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import numpy as np
from nets.SpectralNorm import SpectralNorm
from nets.resnet101 import my_resnet101

class ResnetBlock(nn.Module):
	def __init__(self, in_dim, out_dim, ks):
		super(ResnetBlock, self).__init__()
		self.conv = nn.Sequential(
				nn.Conv2d(in_dim, out_dim, ks, stride=1, padding=ks//2),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(out_dim, out_dim, ks, stride=1, padding=ks//2)
			)
	
	def forward(self, input):
		conv_out = self.conv(input)
		return  conv_out + input

class ResnetSNBlock(nn.Module):
	def __init__(self, in_dim, out_dim, ks):
		super(ResnetSNBlock, self).__init__()
		self.conv = nn.Sequential(
				SpectralNorm(nn.Conv2d(in_dim, out_dim, ks, stride=1, padding=ks//2)),
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(out_dim, out_dim, ks, stride=1, padding=ks//2))
			)
	
	def forward(self, input):
		conv_out = self.conv(input)
		return  conv_out + input


############################################### frame det disc ###################################
class FrameDetDiscriminator(nn.Module):
	def __init__(self, args):
		super(FrameDetDiscriminator, self).__init__()
		self.args=args
		self.input_dim = 23 if self.args.seg_disc else 3
		self.layer = nn.Sequential(
				nn.Conv2d(self.input_dim, 16, 3, 1, 1),
				nn.LeakyReLU(0.2, inplace=True),

				# nn.Conv2d(32, 32, 3, 1, 1),
				# nn.BatchNorm2d(32),
				# nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(16, 16, 3, 1, 1),
				nn.BatchNorm2d(16),
				nn.LeakyReLU(0.2, inplace=True),            # 32x64x64

				nn.Conv2d(16, 32, 3, 2, 1),
				nn.BatchNorm2d(32),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(32, 32, 3, 1, 1),
				nn.BatchNorm2d(32),
				nn.LeakyReLU(0.2, inplace=True),                # 64x32x32

				nn.Conv2d(32, 64, 3, 2, 1),
				nn.BatchNorm2d(64),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(64, 64, 3, 1, 1),
				nn.BatchNorm2d(64),
				nn.LeakyReLU(0.2, inplace=True),                # 96x16x16

				nn.Conv2d(64, 96, 3, 2, 1),
				nn.BatchNorm2d(96),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(96, 96, 3, 1, 1),
				nn.BatchNorm2d(96),
				nn.LeakyReLU(0.2, inplace=True),                # 128x8x8

				# nn.Conv2d(128, 64, 3, 1, 1),
				# nn.BatchNorm2d(64),
				# nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(96, 1, 3, 1, 1),
				nn.AvgPool2d(8)                                 # 1x1x1
			)
		self.H = 64
		self.W = 64

	def forward(self, x, seg, bboxes):
		'''
			bboxes (hlt, wlt, hrb, wrb) shape (bs, 3, 4, 4)
		'''
		bs = x.size(0)
		input = x
		results = []
		bboxes = bboxes[:,1, 1:] # only choose mid frame
		assert bboxes.size(1) == 4
		if self.args.seg_disc:
			input = torch.cat([x, seg], dim=1)
		patches = []
		for i in range(bs):
			for j in range(4):
				box = bboxes[i,j]
				assert box.sum() != 0
				patch = input[i, :, int(box[0]):int(box[2])+1, int(box[1]):int(box[3])+1]
				patch = F.interpolate(patch.unsqueeze(0), size=(self.H, self.W), mode='bilinear', align_corners=True)
				patches.append(patch)
		patches = torch.cat(patches, dim=0) # (bs*4,c,64,64)
		result = self.layer(patches)
		result=result.view(bs, 4)
		result = result.mean(dim=1,keepdim=True)
		return result


class FrameSNDetDiscriminator(nn.Module):
	def __init__(self, args):
		super(FrameSNDetDiscriminator, self).__init__()
		self.args=args
		self.input_dim = 23 if self.args.seg_disc else 3
		self.layer = nn.Sequential(
				SpectralNorm(nn.Conv2d(self.input_dim, 16, 3, 1, 1)),
				nn.LeakyReLU(0.2, inplace=True),

				# nn.Conv2d(32, 32, 3, 1, 1),
				# nn.BatchNorm2d(32),
				# nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(16, 16, 3, 1, 1)),
				nn.LeakyReLU(0.2, inplace=True),                # 32x64x64

				SpectralNorm(nn.Conv2d(16, 32, 3, 2, 1)),
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(32, 32, 3, 1, 1)),
				nn.LeakyReLU(0.2, inplace=True),                # 64x32x32

				SpectralNorm(nn.Conv2d(32, 64, 3, 2, 1)),
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(64, 64, 3, 1, 1)),
				nn.LeakyReLU(0.2, inplace=True),                # 96x16x16

				SpectralNorm(nn.Conv2d(64, 96, 3, 2, 1)),
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(96, 96, 3, 1, 1)),
				nn.LeakyReLU(0.2, inplace=True),                # 128x8x8

				# nn.Conv2d(128, 64, 3, 1, 1),
				# nn.BatchNorm2d(64),
				# nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(96, 1, 3, 1, 1)),
				nn.AvgPool2d(8)                                 # 1x1x1
			)
		self.H = 64
		self.W = 64

	def forward(self, x, seg, bboxes):
		'''
			bboxes (hlt, wlt, hrb, wrb) shape (bs, 3, 4, 4)
		'''
		bs = x.size(0)
		TRACK_NUM = self.args.num_track_per_img
		input = x
		results = []
		bboxes = bboxes[:,1] # only choose mid frame
		assert bboxes.size(1) == TRACK_NUM
		if self.args.seg_disc:
			input = torch.cat([x, seg], dim=1)
		patches = []
		for i in range(bs):
			for j in range(TRACK_NUM):
				box = bboxes[i,j, 1:]
				assert box.sum() != 0
				patch = input[i, :, int(box[0]):int(box[2])+1, int(box[1]):int(box[3])+1]
				patch = F.interpolate(patch.unsqueeze(0), size=(self.H, self.W), mode='bilinear', align_corners=True)
				patches.append(patch)
		patches = torch.cat(patches, dim=0) # (bs*4,c,64,64)
		result = self.layer(patches)
		result=result.view(bs, TRACK_NUM)
		# result.clamp_(-1,1)
		result = result.mean(dim=1,keepdim=True)
		return result


class FrameLSSNDetDiscriminator(nn.Module):
	def __init__(self, args):
		super(FrameLSSNDetDiscriminator, self).__init__()
		self.args=args
		self.input_dim = 23 if self.args.seg_disc else 3
		self.layer = nn.Sequential(
				SpectralNorm(nn.Conv2d(self.input_dim, 64, 3, 1, 1)),
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(64, 64, 3, 1, 1)),
				nn.LeakyReLU(0.2, inplace=True),                # 32x64x64

				SpectralNorm(nn.Conv2d(64, 128, 3, 2, 1)),
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(128, 128, 3, 1, 1)),
				nn.LeakyReLU(0.2, inplace=True),                # 64x32x32

				SpectralNorm(nn.Conv2d(128, 256, 3, 2, 1)),
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(256, 256, 3, 1, 1)),
				nn.LeakyReLU(0.2, inplace=True),                # 96x16x16

				SpectralNorm(nn.Conv2d(256, 256, 3, 2, 1)),
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(256, 256, 3, 1, 1)),
				nn.LeakyReLU(0.2, inplace=True),                # 128x8x8

				# nn.Conv2d(128, 64, 3, 1, 1),
				# nn.BatchNorm2d(64),
				# nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(256, 1, 3, 1, 1)),
				nn.AvgPool2d(8)                                 # 1x1x1
			)
		self.H = 64
		self.W = 64

	def forward(self, x, seg, bboxes):
		'''
			bboxes (hlt, wlt, hrb, wrb) shape (bs, 3, 4, 4)
		'''
		bs = x.size(0)
		TRACK_NUM = self.args.num_track_per_img
		input = x
		results = []
		bboxes = bboxes[:,1] # only choose mid frame
		assert bboxes.size(1) == TRACK_NUM
		if self.args.seg_disc:
			input = torch.cat([x, seg], dim=1)
		patches = []
		for i in range(bs):
			for j in range(TRACK_NUM):
				box = bboxes[i,j, 1:]
				assert box.sum() != 0
				patch = input[i, :, int(box[0]):int(box[2])+1, int(box[1]):int(box[3])+1]
				patch = F.interpolate(patch.unsqueeze(0), size=(self.H, self.W), mode='bilinear', align_corners=True)
				patches.append(patch)
		patches = torch.cat(patches, dim=0) # (bs*4,c,64,64)
		result = self.layer(patches)
		result=result.view(bs, TRACK_NUM)
		# result.clamp_(-1,1)
		result = result.mean(dim=1,keepdim=True)
		return result



###########################################################################
################################## video det disc #########################
###########################################################################
class VideoDetDiscriminator(nn.Module):
	def __init__(self, args):
		super(VideoDetDiscriminator, self).__init__()
		self.args=args
		self.input_dim = 23 if self.args.seg_disc else 3
		self.layer = nn.Sequential(
				nn.Conv2d(3*self.input_dim, 16, 3, 1, 1),
				nn.LeakyReLU(0.2, inplace=True),

				# nn.Conv2d(32, 32, 3, 1, 1),
				# nn.BatchNorm2d(32),
				# nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(16, 16, 3, 1, 1),
				nn.BatchNorm2d(16),
				nn.LeakyReLU(0.2, inplace=True),                # 32x64x64

				nn.Conv2d(16, 32, 3, 2, 1),
				nn.BatchNorm2d(32),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(32, 32, 3, 1, 1),
				nn.BatchNorm2d(32),
				nn.LeakyReLU(0.2, inplace=True),                # 64x32x32

				nn.Conv2d(32, 64, 3, 2, 1),
				nn.BatchNorm2d(64),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(64, 64, 3, 1, 1),
				nn.BatchNorm2d(64),
				nn.LeakyReLU(0.2, inplace=True),                # 96x16x16

				nn.Conv2d(64, 96, 3, 2, 1),
				nn.BatchNorm2d(96),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(96, 96, 3, 1, 1),
				nn.BatchNorm2d(96),
				nn.LeakyReLU(0.2, inplace=True),            # 128x8x8

				# nn.Conv2d(128, 64, 3, 1, 1),
				# nn.BatchNorm2d(64),
				# nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(96, 1, 3, 1, 1),
				nn.AvgPool2d(8)                     # 1x1x1
			)
		self.H = 64
		self.W = 64

	def forward(self, x, seg, input_x, input_seg, bboxes):
		'''
			bboxes (hlt, wlt, hrb, wrb) shape (bs, 3, 4, 4)
		'''
		bs = x.size(0)
		results = []
		cur_img  = torch.cat([x,seg],dim=1) if self.args.seg_disc else x
		for_img  = torch.cat([input_x[:,:3], input_seg[:,:20]], dim=1) if self.args.seg_disc else input_x[:,:3]
		back_img = torch.cat([input_x[:,3:], input_seg[:,20:]], dim=1) if self.args.seg_disc else input_x[:,3:]

		comb_patches = []
		for i in range(bs):
			for j in range(4): # for each box
				mid_box = bboxes[i, 1, j]
				assert mid_box.sum() >0
				cur_patch = cur_img[i, :, mid_box[0]:mid_box[2]+1, mid_box[1]:mid_box[3]+1]
				cur_patch = F.interpolate(cur_patch.unsqueeze(0), size=(self.H, self.W), mode='bilinear', align_corners=True)
				# forward check
				for_box = bboxes[i, 0, j]
				assert for_box.sum() > 0 # for obj exist
				for_patch = for_img[i, :, for_box[0]:for_box[2]+1, for_box[1]:for_box[3]+1]
				for_patch = F.interpolate(for_patch.unsqueeze(0), size=(self.H, self.W), mode='bilinear', align_corners=True)
				# backward check
				back_box = bboxes[i, 2, j]
				assert back_box.sum() > 0 # for obj exist
				back_patch = back_img[i, :, back_box[0]:back_box[2]+1, back_box[1]:back_box[3]+1]
				back_patch = F.interpolate(back_patch.unsqueeze(0), size=(self.H, self.W), mode='bilinear', align_corners=True)
				comb_patches.append(torch.cat([for_patch, cur_patch, back_patch], dim=1))
		comb_patches = torch.cat(comb_patches, dim=0)
		result = self.layer(comb_patches).view(bs, 4).mean(dim=1,keepdim=True)
		return result


class VideoSNDetDiscriminator(nn.Module):
	def __init__(self, args):
		super(VideoSNDetDiscriminator, self).__init__()
		self.args=args
		self.input_dim = 23 if self.args.seg_disc else 3
		self.layer = nn.Sequential(
				SpectralNorm(nn.Conv2d(3*self.input_dim, 16, 3, 1, 1)),
				nn.LeakyReLU(0.2, inplace=True),

				# nn.Conv2d(32, 32, 3, 1, 1),
				# nn.BatchNorm2d(32),
				# nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(16, 16, 3, 1, 1)),
				nn.LeakyReLU(0.2, inplace=True),                # 32x64x64

				SpectralNorm(nn.Conv2d(16, 32, 3, 2, 1)),
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(32, 32, 3, 1, 1)),
				nn.LeakyReLU(0.2, inplace=True),                # 64x32x32

				SpectralNorm(nn.Conv2d(32, 64, 3, 2, 1)),
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(64, 64, 3, 1, 1)),
				nn.LeakyReLU(0.2, inplace=True),                # 96x16x16

				SpectralNorm(nn.Conv2d(64, 96, 3, 2, 1)),
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(96, 96, 3, 1, 1)),
				nn.LeakyReLU(0.2, inplace=True),                # 128x8x8

				# nn.Conv2d(128, 64, 3, 1, 1),
				# nn.BatchNorm2d(64),
				# nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(96, 1, 3, 1, 1)),
				nn.AvgPool2d(8)                     # 1x1x1
			)
		self.H = 64
		self.W = 64

	def forward(self, x, seg, input_x, input_seg, bboxes, sync_neg=False):
		'''
			bboxes (hlt, wlt, hrb, wrb) shape (bs, 3, 4, 4)
		'''
		bs = x.size(0)
		TRACK_NUM = self.args.num_track_per_img
		results = []
		cur_img  = torch.cat([x,seg],dim=1) if self.args.seg_disc else x
		for_img  = torch.cat([input_x[:,:3], input_seg[:,:20]], dim=1) if self.args.seg_disc else input_x[:,:3]
		back_img = torch.cat([input_x[:,3:], input_seg[:,20:]], dim=1) if self.args.seg_disc else input_x[:,3:]

		comb_patches = []
		for i in range(bs):
			for j in range(TRACK_NUM): # for each box
				# flip_flag = np.random.randint(2)
				false_box_index = j
				if sync_neg: # choose other bboxes
					false_box_index = (j + np.random.randint(1,TRACK_NUM) )%TRACK_NUM
				mid_box = bboxes[i, 1, false_box_index]
				assert mid_box.sum() >0
				cur_patch = cur_img[i, :, mid_box[0]:mid_box[2]+1, mid_box[1]:mid_box[3]+1]
				# if sync_neg and flip_flag:
				# 	cur_patch = cur_patch.flip(dims=[1])
				cur_patch = F.interpolate(cur_patch.unsqueeze(0), size=(self.H, self.W), mode='bilinear', align_corners=True)
				# forward check
				for_box = bboxes[i, 0, j]
				assert for_box.sum() > 0 # for obj exist
				for_patch = for_img[i, :, for_box[0]:for_box[2]+1, for_box[1]:for_box[3]+1]
				for_patch = F.interpolate(for_patch.unsqueeze(0), size=(self.H, self.W), mode='bilinear', align_corners=True)
				# backward check
				back_box = bboxes[i, 2, j]
				assert back_box.sum() > 0 # for obj exist
				back_patch = back_img[i, :, back_box[0]:back_box[2]+1, back_box[1]:back_box[3]+1]
				back_patch = F.interpolate(back_patch.unsqueeze(0), size=(self.H, self.W), mode='bilinear', align_corners=True)
				comb_patches.append(torch.cat([for_patch, cur_patch, back_patch], dim=1))
		comb_patches = torch.cat(comb_patches, dim=0)
		result = self.layer(comb_patches).view(bs, TRACK_NUM)
		# result.clamp_(-1,1)
		result = result.mean(dim=1,keepdim=True)
		return result


class VideoLSSNDetDiscriminator(nn.Module):
	def __init__(self, args):
		super(VideoLSSNDetDiscriminator, self).__init__()
		self.args=args
		self.input_dim = 23 if self.args.seg_disc else 3
		self.layer = nn.Sequential(
				SpectralNorm(nn.Conv2d(3*self.input_dim, 64, 3, 1, 1)), # 1, 3
				nn.LeakyReLU(0.2, inplace=True),

				SpectralNorm(nn.Conv2d(64, 64, 3, 1, 1)),				# 1, 5
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(64, 64, 3, 1, 1)),				# 1, 7
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(64, 64, 3, 1, 1)),				# 1, 9
				nn.LeakyReLU(0.2, inplace=True),                # 32x64x64	

				SpectralNorm(nn.Conv2d(64, 128, 3, 2, 1)),				# 2, 11
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(128, 128, 3, 1, 1)),				# 2, 15
				nn.LeakyReLU(0.2, inplace=True),                
				SpectralNorm(nn.Conv2d(128, 128, 3, 1, 1)),				# 2, 19
				nn.LeakyReLU(0.2, inplace=True),                # 64x32x32

				SpectralNorm(nn.Conv2d(128, 256, 3, 2, 1)),				# 4, 23
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(256, 256, 3, 1, 1)),				# 4, 31
				nn.LeakyReLU(0.2, inplace=True),                
				SpectralNorm(nn.Conv2d(256, 256, 3, 1, 1)),				# 4, 39
				nn.LeakyReLU(0.2, inplace=True),                # 96x16x16

				SpectralNorm(nn.Conv2d(256, 256, 3, 2, 1)),				# 8, 47
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(256, 256, 3, 1, 1)),				# 8, 63
				nn.LeakyReLU(0.2, inplace=True),                
				SpectralNorm(nn.Conv2d(256, 256, 3, 1, 1)),				# 8, 79
				nn.LeakyReLU(0.2, inplace=True),                # 128x8x8

				SpectralNorm(nn.Conv2d(256, 64, 3, 1, 1)),				#
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(64, 1, 3, 1, 1)),
				nn.AvgPool2d(8)                     # 1x1x1
			)
		self.H = 64
		self.W = 64

	def forward(self, x, seg, input_x, input_seg, bboxes, sync_neg=False, gt_x=None):
		'''
			bboxes (hlt, wlt, hrb, wrb) shape (bs, 3, 4, 4)
		'''
		bs = x.size(0)
		TRACK_NUM = self.args.num_track_per_img
		results = []
		cur_img  = torch.cat([x,seg],dim=1) if self.args.seg_disc else x
		for_img  = torch.cat([input_x[:,:3], input_seg[:,:20]], dim=1) if self.args.seg_disc else input_x[:,:3]
		back_img = torch.cat([input_x[:,3:], input_seg[:,20:]], dim=1) if self.args.seg_disc else input_x[:,3:]

		comb_patches = []
		area_ratios  = []
		for i in range(bs):
			for j in range(TRACK_NUM): # for each box
				false_box_index = j
				if sync_neg: # choose other bboxes
					false_box_index = (j + np.random.randint(1,TRACK_NUM) )%TRACK_NUM
				mid_box = bboxes[i, 1, false_box_index]
				area_ratios.append(mid_box[0])
				assert mid_box.sum() >0, [ for_box, mid_box, back_box ]
				cur_patch = cur_img[i, :, int(mid_box[1]):int(mid_box[3])+1, int(mid_box[2]):int(mid_box[4])+1]
				cur_patch = F.interpolate(cur_patch.unsqueeze(0), size=(self.H, self.W), mode='bilinear', align_corners=True)
				# forward check
				for_box = bboxes[i, 0, j]
				assert for_box.sum() > 0, [ for_box, mid_box, back_box ] # for obj exist
				for_patch = for_img[i, :, int(for_box[1]):int(for_box[3])+1, int(for_box[2]):int(for_box[4])+1]
				for_patch = F.interpolate(for_patch.unsqueeze(0), size=(self.H, self.W), mode='bilinear', align_corners=True)
				# backward check
				back_box = bboxes[i, 2, j]
				assert back_box.sum() > 0, [ for_box, mid_box, back_box ] # for obj exist
				back_patch = back_img[i, :, int(back_box[1]):int(back_box[3])+1, int(back_box[2]):int(back_box[4])+1]
				back_patch = F.interpolate(back_patch.unsqueeze(0), size=(self.H, self.W), mode='bilinear', align_corners=True)
				comb_patches.append(torch.cat([for_patch, cur_patch, back_patch], dim=1))

		comb_patches = torch.cat(comb_patches, dim=0)
		area_ratios = torch.tensor(area_ratios).cuda(comb_patches.get_device()).view(bs,TRACK_NUM)
		area_ratios = area_ratios/area_ratios.sum(dim=1, keepdim=True)
		result = self.layer(comb_patches).view(bs, TRACK_NUM)
		result = result*area_ratios
		# result.clamp_(-1,1)
		result = result.mean(dim=1,keepdim=True)
		return result


class VideoVecSNDetDiscriminator(nn.Module):
	def __init__(self, args):
		super(VideoVecSNDetDiscriminator, self).__init__()
		self.args=args
		self.input_dim = 23 if self.args.seg_disc else 3
		# resnet101 = torchvision.models.resnet101(pretrained=True)
		# self.resnet = my_resnet101(resnet101) # 2048*2*4
		# for p in self.resnet.parameters():
		# 	p.requires_grad = False
		self.feature_layer = nn.Sequential(
				nn.Conv2d(3, 16, 3, 1, 1),
				nn.LeakyReLU(0.2, inplace=True),

				nn.Conv2d(16, 16, 3, 1, 1),
				nn.BatchNorm2d(16),
				nn.LeakyReLU(0.2, inplace=True),                # 16x64x64

				nn.Conv2d(16, 32, 3, 2, 1),
				nn.BatchNorm2d(32),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(32, 32, 3, 1, 1),
				nn.BatchNorm2d(32),
				nn.LeakyReLU(0.2, inplace=True),                # 32x32x32

				nn.Conv2d(32, 64, 3, 2, 1),
				nn.BatchNorm2d(64),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(64, 64, 3, 1, 1),
				nn.BatchNorm2d(64),
				nn.LeakyReLU(0.2, inplace=True),                # 64x16x16

				nn.Conv2d(64, 96, 3, 2, 1),
				nn.BatchNorm2d(96),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(96, 96, 3, 1, 1),
				nn.BatchNorm2d(96),
				nn.LeakyReLU(0.2, inplace=True),            	# 96x8x8

				nn.Conv2d(96, 128, 3, 2, 1),
				nn.BatchNorm2d(128),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(128, 128, 3, 1, 1),
				nn.BatchNorm2d(128),
				nn.LeakyReLU(0.2, inplace=True),            	# 128x4x4

				nn.Conv2d(128, 256, 3, 2, 1),
				nn.BatchNorm2d(256),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(256, 256, 3, 1, 1)					# 256x2x2		
			)
		self.fc_layer = nn.Linear(1024, 1024) # 1024 vector 
		self.compare_layer = nn.Sequential(
				nn.Linear(1024*3, 512),
				nn.BatchNorm1d(512),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Linear(512, 64),
				nn.BatchNorm1d(64),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Linear(64, 1)
			)

		self.H = 64
		self.W = 64

	def forward(self, x, seg, input_x, input_seg, bboxes, sync_neg=False):
		'''
			bboxes (hlt, wlt, hrb, wrb) shape (bs, 3, 4, 4)
		'''
		bs = x.size(0)
		TRACK_NUM = self.args.num_track_per_img
		results = []
		cur_img  = torch.cat([x,seg],dim=1) if self.args.seg_disc else x
		for_img  = torch.cat([input_x[:,:3], input_seg[:,:20]], dim=1) if self.args.seg_disc else input_x[:,:3]
		back_img = torch.cat([input_x[:,3:], input_seg[:,20:]], dim=1) if self.args.seg_disc else input_x[:,3:]

		comb_patches = []
		for i in range(bs):
			for j in range(TRACK_NUM): # for each box
				false_box_index = j
				mid_box = bboxes[i, 1, false_box_index]
				assert mid_box.sum() >0
				cur_patch = cur_img[i, :, mid_box[0]:mid_box[2]+1, mid_box[1]:mid_box[3]+1]
				if sync_neg:
					cur_patch = cur_patch.flip(dims=[1])
				cur_patch = F.interpolate(cur_patch.unsqueeze(0), size=(self.H, self.W), mode='bilinear', align_corners=True)
				# forward check
				for_box = bboxes[i, 0, j]
				assert for_box.sum() > 0 # for obj exist
				for_patch = for_img[i, :, for_box[0]:for_box[2]+1, for_box[1]:for_box[3]+1]
				for_patch = F.interpolate(for_patch.unsqueeze(0), size=(self.H, self.W), mode='bilinear', align_corners=True)
				# backward check
				back_box = bboxes[i, 2, j]
				assert back_box.sum() > 0 # for obj exist
				back_patch = back_img[i, :, back_box[0]:back_box[2]+1, back_box[1]:back_box[3]+1]
				back_patch = F.interpolate(back_patch.unsqueeze(0), size=(self.H, self.W), mode='bilinear', align_corners=True)
				comb_patches.append(for_patch)
				comb_patches.append(cur_patch)
				comb_patches.append(back_patch)
		comb_patches = torch.cat(comb_patches, dim=0) # (bs*track_num*3, c, 64, 128) 
		conv_features = self.feature_layer(comb_patches)
		fc_features = self.fc_layer( conv_features.view(bs*TRACK_NUM*3, 1024) )
		group_fc_features = fc_features.view(bs*TRACK_NUM, 1024*3)
		result = self.compare_layer(group_fc_features).view(bs, TRACK_NUM) 
		result = result.mean(dim=1,keepdim=True)
		return result


class VideoPoolSNDetDiscriminator(nn.Module):
	def __init__(self, args):
		super(VideoPoolSNDetDiscriminator, self).__init__()
		self.args=args
		self.input_dim = 23 if self.args.seg_disc else 3
		# resnet101 = torchvision.models.resnet101(pretrained=True)
		# self.resnet = my_resnet101(resnet101) # 2048*2*4
		# for p in self.resnet.parameters():
		# 	p.requires_grad = False
		self.feature_layer = nn.Sequential(
				nn.Conv2d(3, 16, 3, 1, 1),
				nn.LeakyReLU(0.2, inplace=True),

				nn.Conv2d(16, 16, 3, 1, 1),
				nn.BatchNorm2d(16),
				nn.LeakyReLU(0.2, inplace=True),                # 16x64x64

				nn.Conv2d(16, 32, 3, 2, 1),
				nn.BatchNorm2d(32),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(32, 32, 3, 1, 1),
				nn.BatchNorm2d(32),
				nn.LeakyReLU(0.2, inplace=True),                # 32x32x32

				nn.Conv2d(32, 64, 3, 2, 1),
				nn.BatchNorm2d(64),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(64, 64, 3, 1, 1),
				nn.BatchNorm2d(64),
				nn.LeakyReLU(0.2, inplace=True),                # 64x16x16

				nn.Conv2d(64, 96, 3, 2, 1),
				nn.BatchNorm2d(96),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(96, 96, 3, 1, 1),
				nn.BatchNorm2d(96),
				nn.LeakyReLU(0.2, inplace=True),            	# 96x8x8

				nn.Conv2d(96, 128, 3, 2, 1),
				nn.BatchNorm2d(128),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(128, 128, 3, 1, 1),
				nn.BatchNorm2d(128),
				nn.LeakyReLU(0.2, inplace=True)            		# 128x4x4	
			)
		self.compare_layer = nn.Sequential(
				SpectralNorm(nn.Conv2d(128*3, 256, 3, 1, 1)),
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(256, 128, 3, 1, 1)),
				nn.LeakyReLU(0.2, inplace=True),    
				SpectralNorm(nn.Conv2d(128, 64, 3, 1, 1)),
				nn.LeakyReLU(0.2, inplace=True),        
				SpectralNorm(nn.Conv2d(64, 1, 3, 1, 1))        						
			)

		self.H = 64
		self.W = 64

	def forward(self, x, seg, input_x, input_seg, bboxes, sync_neg=False):
		'''
			bboxes (hlt, wlt, hrb, wrb) shape (bs, 3, 4, 4)
		'''
		bs = x.size(0)
		TRACK_NUM = self.args.num_track_per_img
		results = []
		cur_img  = torch.cat([x,seg],dim=1) if self.args.seg_disc else x
		for_img  = torch.cat([input_x[:,:3], input_seg[:,:20]], dim=1) if self.args.seg_disc else input_x[:,:3]
		back_img = torch.cat([input_x[:,3:], input_seg[:,20:]], dim=1) if self.args.seg_disc else input_x[:,3:]

		comb_patches = []
		for i in range(bs):
			for j in range(TRACK_NUM): # for each box
				false_box_index = j
				# if sync_neg:
				# 	false_box_index = (j + np.random.randint(1,TRACK_NUM))%TRACK_NUM
				mid_box = bboxes[i, 1, false_box_index]
				assert mid_box.sum() >0
				cur_patch = cur_img[i, :, mid_box[0]:mid_box[2]+1, mid_box[1]:mid_box[3]+1]
				if sync_neg:
					cur_patch = cur_patch.flip(dims=[1])
				cur_patch = F.interpolate(cur_patch.unsqueeze(0), size=(self.H, self.W), mode='bilinear', align_corners=True)
				# forward check
				for_box = bboxes[i, 0, j]
				assert for_box.sum() > 0 # for obj exist
				for_patch = for_img[i, :, for_box[0]:for_box[2]+1, for_box[1]:for_box[3]+1]
				for_patch = F.interpolate(for_patch.unsqueeze(0), size=(self.H, self.W), mode='bilinear', align_corners=True)
				# backward check
				back_box = bboxes[i, 2, j]
				assert back_box.sum() > 0 # for obj exist
				back_patch = back_img[i, :, back_box[0]:back_box[2]+1, back_box[1]:back_box[3]+1]
				back_patch = F.interpolate(back_patch.unsqueeze(0), size=(self.H, self.W), mode='bilinear', align_corners=True)
				comb_patches.append(for_patch)
				comb_patches.append(cur_patch)
				comb_patches.append(back_patch)
		comb_patches = torch.cat(comb_patches, dim=0) # (bs*track_num*3, c, 64, 128) 
		conv_features = self.feature_layer(comb_patches)
		group_fc_features = fc_features.view(bs*TRACK_NUM, 128*3, self.H/16, self.W/16)
		result = self.compare_layer(group_fc_features).view(bs, TRACK_NUM) 
		result = result.mean(dim=1,keepdim=True)
		return result


class VideoGlobalZeroSNDetDiscriminator(nn.Module):
	def __init__(self, args):
		super(VideoGlobalZeroSNDetDiscriminator, self).__init__()
		self.args=args
		self.input_dim = 23 if self.args.seg_disc else 3
		self.layer = nn.Sequential(
				SpectralNorm(nn.Conv2d(3*self.input_dim, 16, 5, 1, 2)),		# 1, 5
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(16, 16, 5, 1, 2)),					# 1, 9
				nn.LeakyReLU(0.2, inplace=True),							
				SpectralNorm(nn.Conv2d(16, 16, 5, 1, 2)),					# 1, 13
				nn.LeakyReLU(0.2, inplace=True),                # 32x128x128	

				SpectralNorm(nn.Conv2d(16, 32, 5, 2, 2)),					# 2, 17
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(32, 32, 5, 1, 2)),					# 2, 25
				nn.LeakyReLU(0.2, inplace=True),               			
				SpectralNorm(nn.Conv2d(32, 32, 5, 1, 2)),					# 2, 33
				nn.LeakyReLU(0.2, inplace=True),                # 64x64x64

				SpectralNorm(nn.Conv2d(32, 64, 5, 2, 1)),					# 4, 41
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(64, 64, 5, 1, 2)),					# 4, 45
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(64, 64, 5, 1, 2)),					# 4, 61
				nn.LeakyReLU(0.2, inplace=True),                # 96x32x32

				SpectralNorm(nn.Conv2d(64, 128, 5, 2, 2)),					# 8, 77
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(128, 128, 5, 1, 2)),					# 8, 109
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(128, 128, 5, 1, 2)),					# 8, 141
				nn.LeakyReLU(0.2, inplace=True),				# 96*16*16
				

				SpectralNorm(nn.Conv2d(128, 128, 3, 2, 1)),					
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(128, 128, 3, 1, 1)),					
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(128, 128, 3, 1, 1)),					
				nn.LeakyReLU(0.2, inplace=True),                # 128x8x8
				SpectralNorm(nn.Conv2d(128, 128, 3, 1, 1)),					
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(128, 128, 3, 1, 1)),					
				nn.LeakyReLU(0.2, inplace=True),                
				SpectralNorm(nn.Conv2d(128, 128, 3, 1, 1)),					
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(128, 128, 3, 1, 1)),					
				nn.LeakyReLU(0.2, inplace=True),                

				SpectralNorm(nn.Conv2d(128, 1, 3, 1, 1)),
				nn.AvgPool2d(8)                     # 1x1x1
			)
		self.H = 64
		self.W = 64

	def forward(self, x, seg, input_x, input_seg, bboxes, sync_neg=False, gt_x=None):
		'''
			bboxes (hlt, wlt, hrb, wrb) shape (bs, 3, 4, 4)
		'''
		bs = x.size(0)
		img_h = x.size(2)
		img_w = x.size(3)
		TRACK_NUM = self.args.num_track_per_img
		results = []
		cur_img  = torch.cat([x,seg],dim=1) if self.args.seg_disc else x
		for_img  = torch.cat([input_x[:,:3], input_seg[:,:20]], dim=1) if self.args.seg_disc else input_x[:,:3]
		back_img = torch.cat([input_x[:,3:], input_seg[:,20:]], dim=1) if self.args.seg_disc else input_x[:,3:]

		comb_patches = []
		for i in range(bs):
			for j in range(TRACK_NUM): # for each box
				# flip_flag = np.random.randint(2)
				false_box_index = j
				if sync_neg: # choose other bboxes
					false_box_index = (j + np.random.randint(1,TRACK_NUM) )%TRACK_NUM
				mid_box = bboxes[i, 1, false_box_index, 1:]
				assert mid_box.sum() > 0
				cur_patch = cur_img[i]
				cur_mask  = torch.zeros((1, img_h, img_w)).cuda(cur_patch.get_device())
				cur_mask[:, int(mid_box[0]):int(mid_box[2])+1, int(mid_box[1]):int(mid_box[3])+1] = 1
				cur_patch = (cur_mask*cur_patch).unsqueeze(0)
				# cur_patch = torch.cat([cur_patch, cur_mask], dim=0).unsqueeze(0)
				# forward check
				for_box = bboxes[i, 0, j, 1:]
				assert for_box.sum() > 0 # for obj exist
				for_patch = for_img[i]
				for_mask  = torch.zeros((1, img_h, img_w)).cuda(for_patch.get_device())
				for_mask[:, int(for_box[0]):int(for_box[2])+1, int(for_box[1]):int(for_box[3])+1] = 1
				for_patch = (for_mask*for_patch).unsqueeze(0)
				# for_patch = torch.cat([for_patch, for_mask], dim=0).unsqueeze(0)

				# backward check
				back_box = bboxes[i, 2, j, 1:]
				assert back_box.sum() > 0 # for obj exist
				back_patch = back_img[i]
				back_mask  = torch.zeros((1, img_h, img_w)).cuda(back_patch.get_device())
				back_mask[:, int(back_box[0]):int(back_box[2])+1, int(back_box[1]):int(back_box[3])+1] = 1
				back_patch = (back_mask*back_patch).unsqueeze(0)
				# back_patch = torch.cat([back_patch, back_mask], dim=0).unsqueeze(0)
				comb_patches.append(torch.cat([for_patch, cur_patch, back_patch], dim=1))
		comb_patches = torch.cat(comb_patches, dim=0)
		result = self.layer(comb_patches).view(bs, TRACK_NUM)
		# result.clamp_(-1,1)
		result = result.mean(dim=1,keepdim=True)
		return result


class VideoGlobalMaskSNDetDiscriminator(nn.Module):
	def __init__(self, args):
		super(VideoGlobalMaskSNDetDiscriminator, self).__init__()
		self.args=args
		self.input_dim = 23 if self.args.seg_disc else 3
		self.input_dim+=1
		self.layer = nn.Sequential(
				SpectralNorm(nn.Conv2d(3*self.input_dim, 16, 5, 1, 2)),		# 1, 5
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(16, 16, 5, 1, 2)),					# 1, 9
				nn.LeakyReLU(0.2, inplace=True),							
				SpectralNorm(nn.Conv2d(16, 16, 5, 1, 2)),					# 1, 13
				nn.LeakyReLU(0.2, inplace=True),                # 32x128x128	

				SpectralNorm(nn.Conv2d(16, 32, 5, 2, 2)),					# 2, 17
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(32, 32, 5, 1, 2)),					# 2, 25
				nn.LeakyReLU(0.2, inplace=True),               			
				SpectralNorm(nn.Conv2d(32, 32, 5, 1, 2)),					# 2, 33
				nn.LeakyReLU(0.2, inplace=True),                # 64x64x64

				SpectralNorm(nn.Conv2d(32, 64, 5, 2, 1)),					# 4, 41
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(64, 64, 5, 1, 2)),					# 4, 45
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(64, 64, 5, 1, 2)),					# 4, 61
				nn.LeakyReLU(0.2, inplace=True),                # 96x32x32

				SpectralNorm(nn.Conv2d(64, 128, 5, 2, 2)),					# 8, 77
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(128, 128, 5, 1, 2)),					# 8, 109
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(128, 128, 5, 1, 2)),					# 8, 141
				nn.LeakyReLU(0.2, inplace=True),				# 96*16*16
				

				SpectralNorm(nn.Conv2d(128, 128, 3, 2, 1)),					
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(128, 128, 3, 1, 1)),					
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(128, 128, 3, 1, 1)),					
				nn.LeakyReLU(0.2, inplace=True),                # 128x8x8
				SpectralNorm(nn.Conv2d(128, 128, 3, 1, 1)),					
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(128, 128, 3, 1, 1)),					
				nn.LeakyReLU(0.2, inplace=True),                
				SpectralNorm(nn.Conv2d(128, 128, 3, 1, 1)),					
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(128, 128, 3, 1, 1)),					
				nn.LeakyReLU(0.2, inplace=True),                

				SpectralNorm(nn.Conv2d(128, 1, 3, 1, 1)),
				nn.AvgPool2d(8)                     # 1x1x1
			)
		self.H = 64
		self.W = 64

	def forward(self, x, seg, input_x, input_seg, bboxes, sync_neg=False, gt_x=None):
		'''
			bboxes (hlt, wlt, hrb, wrb) shape (bs, 3, 4, 4)
		'''
		bs = x.size(0)
		img_h = x.size(2)
		img_w = x.size(3)
		TRACK_NUM = self.args.num_track_per_img
		results = []
		cur_img  = torch.cat([x,seg],dim=1) if self.args.seg_disc else x
		gt_img   = gt_x
		for_img  = torch.cat([input_x[:,:3], input_seg[:,:20]], dim=1) if self.args.seg_disc else input_x[:,:3]
		back_img = torch.cat([input_x[:,3:], input_seg[:,20:]], dim=1) if self.args.seg_disc else input_x[:,3:]

		comb_patches = []
		for i in range(bs):
			for j in range(TRACK_NUM): # for each box
				# flip_flag = np.random.randint(2)
				false_box_index = j
				if sync_neg: # choose other bboxes
					false_box_index = (j + np.random.randint(1,TRACK_NUM) )%TRACK_NUM
				mid_box = bboxes[i, 1, false_box_index, 1:]
				assert mid_box.sum() > 0
				cur_patch = cur_img[i]
				cur_gt_patch = gt_img[i].clone()
				cur_gt_patch[:, int(mid_box[0]):int(mid_box[2])+1, int(mid_box[1]):int(mid_box[3])+1] = \
					cur_patch[:, int(mid_box[0]):int(mid_box[2])+1, int(mid_box[1]):int(mid_box[3])+1]
				cur_mask  = torch.zeros((1, img_h, img_w)).cuda(cur_patch.get_device())
				cur_mask[:, int(mid_box[0]):int(mid_box[2])+1, int(mid_box[1]):int(mid_box[3])+1] = 1
				# cur_patch = (cur_mask*cur_patch).unsqueeze(0)
				cur_patch = torch.cat([cur_gt_patch, cur_mask], dim=0).unsqueeze(0)
				# forward check
				for_box = bboxes[i, 0, j, 1:]
				assert for_box.sum() > 0 # for obj exist
				for_patch = for_img[i]
				for_mask  = torch.zeros((1, img_h, img_w)).cuda(for_patch.get_device())
				for_mask[:, int(for_box[0]):int(for_box[2])+1, int(for_box[1]):int(for_box[3])+1] = 1
				# for_patch = (for_mask*for_patch).unsqueeze(0)
				for_patch = torch.cat([for_patch, for_mask], dim=0).unsqueeze(0)

				# backward check
				back_box = bboxes[i, 2, j, 1:]
				assert back_box.sum() > 0 # for obj exist
				back_patch = back_img[i]
				back_mask  = torch.zeros((1, img_h, img_w)).cuda(back_patch.get_device())
				back_mask[:, int(back_box[0]):int(back_box[2])+1, int(back_box[1]):int(back_box[3])+1] = 1
				# back_patch = (back_mask*back_patch).unsqueeze(0)
				back_patch = torch.cat([back_patch, back_mask], dim=0).unsqueeze(0)

				comb_patches.append(torch.cat([for_patch, cur_patch, back_patch], dim=1))
		comb_patches = torch.cat(comb_patches, dim=0)
		result = self.layer(comb_patches).view(bs, TRACK_NUM)
		# result.clamp_(-1,1)
		result = result.mean(dim=1,keepdim=True)
		return result


class VideoGlobalCoordSNDetDiscriminator(nn.Module):
	def __init__(self, args):
		super(VideoGlobalCoordSNDetDiscriminator, self).__init__()
		self.args=args
		self.input_dim = 23 if self.args.seg_disc else 3
		self.input_dim+=2
		self.layer = nn.Sequential(
				SpectralNorm(nn.Conv2d(3*self.input_dim, 64, 3, 1, 1)),
				nn.LeakyReLU(0.2, inplace=True),

				SpectralNorm(nn.Conv2d(64, 64, 3, 1, 1)),
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(64, 64, 3, 1, 1)),
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(64, 64, 3, 1, 1)),
				nn.LeakyReLU(0.2, inplace=True),                # 32x64x64

				SpectralNorm(nn.Conv2d(64, 128, 3, 2, 1)),
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(128, 128, 3, 1, 1)),
				nn.LeakyReLU(0.2, inplace=True),                
				SpectralNorm(nn.Conv2d(128, 128, 3, 1, 1)),
				nn.LeakyReLU(0.2, inplace=True),                # 64x32x32

				SpectralNorm(nn.Conv2d(128, 256, 3, 2, 1)),
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(256, 256, 3, 1, 1)),
				nn.LeakyReLU(0.2, inplace=True),                
				SpectralNorm(nn.Conv2d(256, 256, 3, 1, 1)),
				nn.LeakyReLU(0.2, inplace=True),                # 96x16x16

				SpectralNorm(nn.Conv2d(256, 256, 3, 2, 1)),
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(256, 256, 3, 1, 1)),
				nn.LeakyReLU(0.2, inplace=True),                
				SpectralNorm(nn.Conv2d(256, 256, 3, 1, 1)),
				nn.LeakyReLU(0.2, inplace=True),                # 128x8x8

				SpectralNorm(nn.Conv2d(256, 64, 3, 1, 1)),
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(64, 1, 3, 1, 1)),
				nn.AvgPool2d(8)                     # 1x1x1
			)
		self.H = 64
		self.W = 64

		img_h = self.args.input_h
		img_w = self.args.input_w
		w_t = torch.matmul(
			torch.ones(img_h, 1), torch.linspace(-1.0, 1.0, img_w).view(1, img_w)).unsqueeze(0)
		h_t = torch.matmul(
			torch.linspace(-1.0, 1.0, img_h).view(img_h, 1), torch.ones(1, img_w)).unsqueeze(0)
		self.img_coord = torch.cat([h_t, w_t], dim=0).cuda(self.args.rank) #(2, h, w)

	def forward(self, x, seg, input_x, input_seg, bboxes, sync_neg=False, gt_x=None):
		'''
			bboxes (hlt, wlt, hrb, wrb) shape (bs, 3, 4, 4)
		'''
		bs = x.size(0)
		TRACK_NUM = self.args.num_track_per_img
		for_base_img_coord = self.img_coord.repeat(bs, 1, 1, 1)
		back_base_img_coord = self.img_coord.repeat(bs, 1, 1, 1)
		cur_base_img_coord = self.img_coord.repeat(bs, 1, 1, 1)

		for_coord_img  = torch.cat([for_base_img_coord, input_x[:,:3], input_seg[:,:20]], dim=1) if self.args.seg_disc else \
								torch.cat([for_base_img_coord, input_x[:,:3]], dim=1)
		back_coord_img = torch.cat([back_base_img_coord, input_x[:,3:6], input_seg[:,20:40]], dim=1) if self.args.seg_disc else \
								torch.cat([back_base_img_coord, input_x[:,3:6]], dim=1)
		cur_coord_img = torch.cat([cur_base_img_coord, x, seg], dim=1) if self.args.seg_disc else \
								torch.cat([cur_base_img_coord, x], dim=1)

		results = []

		comb_patches = []
		area_ratios  = []
		for i in range(bs):
			for j in range(TRACK_NUM): # for each box
				false_box_index = j
				if sync_neg: # choose other bboxes
					false_box_index = (j + np.random.randint(1,TRACK_NUM) )%TRACK_NUM
				mid_box = bboxes[i, 1, false_box_index]
				area_ratios.append(mid_box[0])
				assert mid_box.sum() >0, [ for_box, mid_box, back_box ]
				cur_patch = cur_coord_img[i, :, int(mid_box[1]):int(mid_box[3])+1, int(mid_box[2]):int(mid_box[4])+1]
				cur_patch = F.interpolate(cur_patch.unsqueeze(0), size=(self.H, self.W), mode='bilinear', align_corners=True)
				# forward check
				for_box = bboxes[i, 0, j]
				assert for_box.sum() > 0, [ for_box, mid_box, back_box ] # for obj exist
				for_patch = for_coord_img[i, :, int(for_box[1]):int(for_box[3])+1, int(for_box[2]):int(for_box[4])+1]
				for_patch = F.interpolate(for_patch.unsqueeze(0), size=(self.H, self.W), mode='bilinear', align_corners=True)
				# backward check
				back_box = bboxes[i, 2, j]
				assert back_box.sum() > 0, [ for_box, mid_box, back_box ] # for obj exist
				back_patch = back_coord_img[i, :, int(back_box[1]):int(back_box[3])+1, int(back_box[2]):int(back_box[4])+1]
				back_patch = F.interpolate(back_patch.unsqueeze(0), size=(self.H, self.W), mode='bilinear', align_corners=True)
				comb_patches.append(torch.cat([for_patch, cur_patch, back_patch], dim=1))

		comb_patches = torch.cat(comb_patches, dim=0)
		area_ratios = torch.tensor(area_ratios).cuda(comb_patches.get_device()).view(bs,TRACK_NUM)
		area_ratios = area_ratios/area_ratios.sum(dim=1, keepdim=True)
		result = self.layer(comb_patches).view(bs, TRACK_NUM)
		result = result*area_ratios
		# result.clamp_(-1,1)
		result = result.mean(dim=1,keepdim=True)
		return result


class VideoGlobalResSNDetDiscriminator(nn.Module):
	def __init__(self, args):
		super(VideoGlobalResSNDetDiscriminator, self).__init__()
		self.args=args
		self.input_dim = 23 if self.args.seg_disc else 3
		self.layer = nn.Sequential(
				SpectralNorm(nn.Conv2d(3*self.input_dim, 64, 3, 1, 1)),		# 1
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(64, 64, 3, 1, 1)),					# 1
				nn.LeakyReLU(0.2, inplace=True),						
				ResnetSNBlock(64, 64, 3),
				ResnetSNBlock(64, 64, 3),									# 32x128x128	

				SpectralNorm(nn.Conv2d(64, 128, 3, 2, 1)),					# 2
				nn.LeakyReLU(0.2, inplace=True),
				ResnetSNBlock(128, 128, 3),
				ResnetSNBlock(128, 128, 3),									# 64x32x32

				SpectralNorm(nn.Conv2d(128, 128, 3, 2, 1)),					# 4
				nn.LeakyReLU(0.2, inplace=True),
				ResnetSNBlock(128, 128, 3),
				ResnetSNBlock(128, 128, 3),               					# 96x16x16

				SpectralNorm(nn.Conv2d(128, 128, 3, 2, 1)),					# 8
				nn.LeakyReLU(0.2, inplace=True),
				ResnetSNBlock(128, 128, 3),
				ResnetSNBlock(128, 128, 3),  									

				SpectralNorm(nn.Conv2d(128, 256, 3, 2, 1)),					# 8
				nn.LeakyReLU(0.2, inplace=True),
				ResnetSNBlock(256, 256, 3),
				ResnetSNBlock(256, 256, 3),               					# 128x8x8

				SpectralNorm(nn.Conv2d(256, 1, 3, 1, 1)),
				nn.AvgPool2d(8)                     # 1x1x1
			)
		self.H = 64
		self.W = 64

	def forward(self, x, seg, input_x, input_seg, bboxes, sync_neg=False):
		'''
			bboxes (hlt, wlt, hrb, wrb) shape (bs, 3, 4, 4)
		'''
		bs = x.size(0)
		img_h = x.size(2)
		img_w = x.size(3)
		TRACK_NUM = self.args.num_track_per_img
		results = []
		cur_img  = torch.cat([x,seg],dim=1) if self.args.seg_disc else x
		for_img  = torch.cat([input_x[:,:3], input_seg[:,:20]], dim=1) if self.args.seg_disc else input_x[:,:3]
		back_img = torch.cat([input_x[:,3:], input_seg[:,20:]], dim=1) if self.args.seg_disc else input_x[:,3:]

		comb_patches = []
		for i in range(bs):
			for j in range(TRACK_NUM): # for each box
				# flip_flag = np.random.randint(2)
				false_box_index = j
				if sync_neg: # choose other bboxes
					false_box_index = (j + np.random.randint(1,TRACK_NUM) )%TRACK_NUM
				mid_box = bboxes[i, 1, false_box_index, 1:]
				assert mid_box.sum() > 0
				cur_patch = cur_img[i]
				cur_mask  = torch.zeros((1, img_h, img_w)).cuda(cur_patch.get_device())
				cur_mask[:, int(mid_box[0]):int(mid_box[2])+1, int(mid_box[1]):int(mid_box[3])+1] = 1
				cur_patch = (cur_mask*cur_patch).unsqueeze(0)
				# cur_patch = torch.cat([cur_patch, cur_mask], dim=0).unsqueeze(0)
				# forward check
				for_box = bboxes[i, 0, j, 1:]
				assert for_box.sum() > 0 # for obj exist
				for_patch = for_img[i]
				for_mask  = torch.zeros((1, img_h, img_w)).cuda(for_patch.get_device())
				for_mask[:, int(for_box[0]):int(for_box[2])+1, int(for_box[1]):int(for_box[3])+1] = 1
				for_patch = (for_mask*for_patch).unsqueeze(0)
				# for_patch = torch.cat([for_patch, for_mask], dim=0).unsqueeze(0)

				# backward check
				back_box = bboxes[i, 2, j, 1:]
				assert back_box.sum() > 0 # for obj exist
				back_patch = back_img[i]
				back_mask  = torch.zeros((1, img_h, img_w)).cuda(back_patch.get_device())
				back_mask[:, int(back_box[0]):int(back_box[2])+1, int(back_box[1]):int(back_box[3])+1] = 1
				back_patch = (back_mask*back_patch).unsqueeze(0)
				# back_patch = torch.cat([back_patch, back_mask], dim=0).unsqueeze(0)
				comb_patches.append(torch.cat([for_patch, cur_patch, back_patch], dim=1))
		comb_patches = torch.cat(comb_patches, dim=0)
		result = self.layer(comb_patches).view(bs, TRACK_NUM)
		# result.clamp_(-1,1)
		result = result.mean(dim=1,keepdim=True)
		return result

class VideoLocalPatchSNDetDiscriminator(nn.Module):
	def __init__(self, args):
		super(VideoLocalPatchSNDetDiscriminators, self).__init__()
		self.args=args
		self.input_dim = 3
		self.net = nn.Sequential(
				nn.Conv2d(3*self.input_dim, 64, 3, 2, 1),					# 2, 3
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(64, 64, 3, 1, 1)),					# 2, 7
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(64, 64, 3, 1, 1)),					# 2, 11		
				nn.LeakyReLU(0.2, inplace=True),							# 32x32

				SpectralNorm(nn.Conv2d(64, 128, 3, 2, 1)),					# 4, 15
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(128, 128, 3, 1, 1)),					# 4, 23
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(128, 128, 3, 1, 1)),					# 4, 31
				nn.LeakyReLU(0.2, inplace=True),							# 16x16

				SpectralNorm(nn.Conv2d(128, 256, 3, 2, 1)),					# 8, 39
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(256, 256, 1, 1, 0)),					
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(256, 256, 1, 1, 0)),					
																			# 8x8
				SpectralNorm(nn.Conv2d(256, 64, 1, 1, 0)),
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(64, 1, 1, 1, 0)),					
				nn.AvgPool2d(8)
			)

	def forward(self, x, seg, input_x, input_seg, bboxes, sync_neg=False, gt_x=None):
		'''
			bboxes (hlt, wlt, hrb, wrb) shape (bs, 3, 4, 4)
		'''
		bs = x.size(0)
		TRACK_NUM = self.args.num_track_per_img
		results = []
		cur_img  = torch.cat([x,seg],dim=1) if self.args.seg_disc else x
		for_img  = torch.cat([input_x[:,:3], input_seg[:,:20]], dim=1) if self.args.seg_disc else input_x[:,:3]
		back_img = torch.cat([input_x[:,3:], input_seg[:,20:]], dim=1) if self.args.seg_disc else input_x[:,3:]

		comb_patches = []
		area_ratios  = []
		for i in range(bs):
			for j in range(TRACK_NUM): # for each box
				false_box_index = j
				if sync_neg: # choose other bboxes
					false_box_index = (j + np.random.randint(1,TRACK_NUM) )%TRACK_NUM
				mid_box = bboxes[i, 1, false_box_index]
				area_ratios.append(mid_box[0])
				assert mid_box.sum() >0, [ for_box, mid_box, back_box ]
				cur_patch = cur_img[i, :, int(mid_box[1]):int(mid_box[3])+1, int(mid_box[2]):int(mid_box[4])+1]
				cur_patch = F.interpolate(cur_patch.unsqueeze(0), size=(self.H, self.W), mode='bilinear', align_corners=True)
				# forward check
				for_box = bboxes[i, 0, j]
				assert for_box.sum() > 0, [ for_box, mid_box, back_box ] # for obj exist
				for_patch = for_img[i, :, int(for_box[1]):int(for_box[3])+1, int(for_box[2]):int(for_box[4])+1]
				for_patch = F.interpolate(for_patch.unsqueeze(0), size=(self.H, self.W), mode='bilinear', align_corners=True)
				# backward check
				back_box = bboxes[i, 2, j]
				assert back_box.sum() > 0, [ for_box, mid_box, back_box ] # for obj exist
				back_patch = back_img[i, :, int(back_box[1]):int(back_box[3])+1, int(back_box[2]):int(back_box[4])+1]
				back_patch = F.interpolate(back_patch.unsqueeze(0), size=(self.H, self.W), mode='bilinear', align_corners=True)
				comb_patches.append(torch.cat([for_patch, cur_patch, back_patch], dim=1))

		comb_patches = torch.cat(comb_patches, dim=0)
		area_ratios = torch.tensor(area_ratios).cuda(comb_patches.get_device()).view(bs,TRACK_NUM)
		area_ratios = area_ratios/area_ratios.sum(dim=1, keepdim=True)
		result = self.layer(comb_patches).view(bs, TRACK_NUM)
		result = result*area_ratios
		# result.clamp_(-1,1)
		result = result.mean(dim=1,keepdim=True)
		return result	