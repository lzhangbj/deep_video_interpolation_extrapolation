import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import numpy as np
from nets.SpectralNorm import SpectralNorm

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
				nn.LeakyReLU(0.2, inplace=True),			# 32x64x64

				nn.Conv2d(16, 32, 3, 2, 1),
				nn.BatchNorm2d(32),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(32, 32, 3, 1, 1),
				nn.BatchNorm2d(32),
				nn.LeakyReLU(0.2, inplace=True),				# 64x32x32

				nn.Conv2d(32, 64, 3, 2, 1),
				nn.BatchNorm2d(64),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(64, 64, 3, 1, 1),
				nn.BatchNorm2d(64),
				nn.LeakyReLU(0.2, inplace=True),				# 96x16x16

				nn.Conv2d(64, 96, 3, 2, 1),
				nn.BatchNorm2d(96),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(96, 96, 3, 1, 1),
				nn.BatchNorm2d(96),
				nn.LeakyReLU(0.2, inplace=True),				# 128x8x8

				# nn.Conv2d(128, 64, 3, 1, 1),
				# nn.BatchNorm2d(64),
				# nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(96, 1, 3, 1, 1),
				nn.AvgPool2d(8)									# 1x1x1
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
		bboxes = bboxes[:,1] # only choose mid frame
		assert bboxes.size(1) == 4
		if self.args.seg_disc:
			input = torch.cat([x, seg], dim=1)
		patches = []
		for i in range(bs):
			for j in range(4):
				box = bboxes[i,j]
				assert box.sum() != 0
				patch = input[i, :, box[0]:box[2]+1, box[1]:box[3]+1]
				patch = F.interpolate(patch.unsqueeze(0), size=(self.H, self.W), mode='bilinear', align_corners=True)
				patches.append(patch)
		patches = torch.cat(patches, dim=0) # (bs*4,c,64,64)
		result = self.layer(patches)
		result=result.view(bs, 4)
		result = result.mean(dim=1,keepdim=True)
		return result


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
				nn.LeakyReLU(0.2, inplace=True),				# 32x64x64

				nn.Conv2d(16, 32, 3, 2, 1),
				nn.BatchNorm2d(32),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(32, 32, 3, 1, 1),
				nn.BatchNorm2d(32),
				nn.LeakyReLU(0.2, inplace=True),				# 64x32x32

				nn.Conv2d(32, 64, 3, 2, 1),
				nn.BatchNorm2d(64),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(64, 64, 3, 1, 1),
				nn.BatchNorm2d(64),
				nn.LeakyReLU(0.2, inplace=True),				# 96x16x16

				nn.Conv2d(64, 96, 3, 2, 1),
				nn.BatchNorm2d(96),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(96, 96, 3, 1, 1),
				nn.BatchNorm2d(96),
				nn.LeakyReLU(0.2, inplace=True),			# 128x8x8

				# nn.Conv2d(128, 64, 3, 1, 1),
				# nn.BatchNorm2d(64),
				# nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(96, 1, 3, 1, 1),
				nn.AvgPool2d(8)						# 1x1x1
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
				nn.LeakyReLU(0.2, inplace=True),				# 32x64x64

				SpectralNorm(nn.Conv2d(16, 32, 3, 2, 1)),
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(32, 32, 3, 1, 1)),
				nn.LeakyReLU(0.2, inplace=True),				# 64x32x32

				SpectralNorm(nn.Conv2d(32, 64, 3, 2, 1)),
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(64, 64, 3, 1, 1)),
				nn.LeakyReLU(0.2, inplace=True),				# 96x16x16

				SpectralNorm(nn.Conv2d(64, 96, 3, 2, 1)),
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(96, 96, 3, 1, 1)),
				nn.LeakyReLU(0.2, inplace=True),				# 128x8x8

				# nn.Conv2d(128, 64, 3, 1, 1),
				# nn.BatchNorm2d(64),
				# nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(96, 1, 3, 1, 1)),
				nn.AvgPool2d(8)									# 1x1x1
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
		bboxes = bboxes[:,1] # only choose mid frame
		assert bboxes.size(1) == 4
		if self.args.seg_disc:
			input = torch.cat([x, seg], dim=1)
		patches = []
		for i in range(bs):
			for j in range(4):
				box = bboxes[i,j]
				assert box.sum() != 0
				patch = input[i, :, box[0]:box[2]+1, box[1]:box[3]+1]
				patch = F.interpolate(patch.unsqueeze(0), size=(self.H, self.W), mode='bilinear', align_corners=True)
				patches.append(patch)
		patches = torch.cat(patches, dim=0) # (bs*4,c,64,64)
		result = self.layer(patches)
		result=result.view(bs, 4)
		# result.clamp_(-1,1)
		result = result.mean(dim=1,keepdim=True)
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
				nn.LeakyReLU(0.2, inplace=True),				# 32x64x64

				SpectralNorm(nn.Conv2d(16, 32, 3, 2, 1)),
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(32, 32, 3, 1, 1)),
				nn.LeakyReLU(0.2, inplace=True),				# 64x32x32

				SpectralNorm(nn.Conv2d(32, 64, 3, 2, 1)),
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(64, 64, 3, 1, 1)),
				nn.LeakyReLU(0.2, inplace=True),				# 96x16x16

				SpectralNorm(nn.Conv2d(64, 96, 3, 2, 1)),
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(96, 96, 3, 1, 1)),
				nn.LeakyReLU(0.2, inplace=True),				# 128x8x8

				# nn.Conv2d(128, 64, 3, 1, 1),
				# nn.BatchNorm2d(64),
				# nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(96, 1, 3, 1, 1)),
				nn.AvgPool2d(8)						# 1x1x1
			)
		self.H = 64
		self.W = 64

	def forward(self, x, seg, input_x, input_seg, bboxes, sync_neg=False):
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
				false_box_index = j
				if sync_neg:
					false_box_index = (j + np.random.randint(1,4))%4
				mid_box = bboxes[i, 1, false_box_index]
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
		result = self.layer(comb_patches).view(bs, 4)
		# result.clamp_(-1,1)
		result = result.mean(dim=1,keepdim=True)
		return result


class VideoLSSNDetDiscriminator(nn.Module):
	def __init__(self, args):
		super(VideoLSSNDetDiscriminator, self).__init__()
		self.args=args
		self.input_dim = 23 if self.args.seg_disc else 3
		self.layer = nn.Sequential(
				SpectralNorm(nn.Conv2d(3*self.input_dim, 32, 3, 1, 1)),
				nn.LeakyReLU(0.2, inplace=True),

				SpectralNorm(nn.Conv2d(32, 32, 3, 1, 1)),
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(32, 32, 3, 1, 1)),
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(32, 32, 3, 1, 1)),
				nn.LeakyReLU(0.2, inplace=True),				# 32x64x64

				SpectralNorm(nn.Conv2d(32, 64, 3, 2, 1)),
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(64, 64, 3, 1, 1)),
				nn.LeakyReLU(0.2, inplace=True),				
				SpectralNorm(nn.Conv2d(64, 64, 3, 1, 1)),
				nn.LeakyReLU(0.2, inplace=True),				# 64x32x32

				SpectralNorm(nn.Conv2d(64, 128, 3, 2, 1)),
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(128, 128, 3, 1, 1)),
				nn.LeakyReLU(0.2, inplace=True),				
				SpectralNorm(nn.Conv2d(128, 128, 3, 1, 1)),
				nn.LeakyReLU(0.2, inplace=True),				# 96x16x16

				SpectralNorm(nn.Conv2d(128, 256, 3, 2, 1)),
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(256, 256, 3, 1, 1)),
				nn.LeakyReLU(0.2, inplace=True),				
				SpectralNorm(nn.Conv2d(256, 256, 3, 1, 1)),
				nn.LeakyReLU(0.2, inplace=True),				# 128x8x8

				SpectralNorm(nn.Conv2d(256, 64, 3, 1, 1)),
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(64, 1, 3, 1, 1)),
				nn.AvgPool2d(8)						# 1x1x1
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
		result = self.layer(comb_patches).view(bs, 4)
		# result.clamp_(-1,1)
		result = result.mean(dim=1,keepdim=True)
		return result