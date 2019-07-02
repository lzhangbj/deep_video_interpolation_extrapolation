import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from utils.data_utils import *
from torch.autograd import Variable as Vb

def preprocess_norm(input_tensor, cuda=True):
	'''

	'''
	mean_arr = torch.tensor([-0.03,-0.088,-0.188])[None,:,None,None]
	std_arr = torch.tensor([0.458,0.448,0.450])[None,:,None,None]
	if cuda:
		device = input_tensor.get_device()
		mean_arr = mean_arr.cuda(device)
		std_arr = std_arr.cuda(device)
	return (input_tensor-mean_arr)/std_arr

def adjust_learning_rate(optimizer, decay=0.1):
	"""Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs"""
	for param_group in optimizer.param_groups:
		param_group['lr'] = decay * param_group['lr']


def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm2d') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)

def transform_seg_one_hot(seg, n_cls, cuda=False):
	'''
		input tensor:
			seg  (bs, h, w) Long Tensors
			seg  (bs, vid, h, w)
		output tensor
			seg_one_hot  (bs, n_cls, h, w) float tensor
			seg_one_hot  (bs, vid, n_cls, h, w) float tensor

	'''
	if len(seg.size()) == 3:
		seg_one_hot = torch.eye(n_cls)[seg.long()].permute(0,3,1,2).contiguous().float()#.cuda().float()#seg.get_device())
	else:
		raise Exception("shape wrong ", seg_one_hot.size())
	if cuda:
		seg_one_hot = seg_one_hot.float().cuda(seg.get_device())
	return seg_one_hot

def vis_seg_mask(seg, n_classes, seg_id=False):
	'''
		mask (bs, c,h,w) into normed rgb (bs, 3,h,w)
		all tensors
	'''
	global color_map
	assert len(seg.size()) == 4
	if not seg_id:
		id_seg = torch.argmax(seg, dim=1)
	else:
		id_seg = seg.squeeze(1).long()
	color_mapp = torch.tensor(color_map)
	rgb_seg = color_mapp[id_seg].permute(0,3,1,2).contiguous().float()
	return rgb_seg/255

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

class FlowWrapper(nn.Module):
	def __init__(self):
		super(FlowWrapper, self).__init__()

	def forward(self, x, flow):
		# flow: (batch size, 2, height, width)
		# x = x.cuda()
		N = x.size()[0]
		H = x.size()[2]
		W = x.size()[3]
		base_grid = torch.zeros([N, H, W, 2])
		linear_points = torch.linspace(-1, 1, W) if W > 1 else torch.Tensor([-1])
		base_grid[:, :, :, 0] = torch.ger(torch.ones(H), linear_points).expand_as(base_grid[:, :, :, 0])
		linear_points = torch.linspace(-1, 1, H) if H > 1 else torch.Tensor([-1])
		base_grid[:, :, :, 1] = torch.ger(linear_points, torch.ones(W)).expand_as(base_grid[:, :, :, 1])
		if x.is_cuda:
			base_grid = Vb(base_grid).cuda()
		else:
			base_grid = Vb(base_grid)
		# print flow.shape
		flow = flow.transpose(1, 2).transpose(2, 3)
		# print flow.size()
		grid = base_grid - flow
		# print grid.size()
		out = F.grid_sample(x, grid)
		return out

def warp(frame, flow, opt, flowwarpper, mask):
	'''Use mask before warpping'''
	out = [torch.unsqueeze(flowwarpper(frame, flow[:, :, i, :, :] * mask[:, i:i + 1, ...]), 1)
		   for i in range(opt.vid_length)]
	output = torch.cat(out, 1)  # (64, 4, 3, 128, 128)
	return output


def warp_back(frame, flowback, opt, flowwarpper, mask):
	prevframe = [
		torch.unsqueeze(flowwarpper(frame[:, ii, :, :, :], -flowback[:, :, ii, :, :] * mask[:, ii:ii + 1, ...]), 1)
		for ii in range(opt.vid_length)]
	output = torch.cat(prevframe, 1)
	return output


def refine(input, flow, mask, refine_net, opt, noise_bg):
	'''Go through the refine network.'''
	# apply mask to the warpped image
	bs, _, _, h, w = input.size()
	if opt.seg:
		noise_seg = noise_bg.new(bs, 20, h, w).fill_(0)
		noise = torch.cat([noise_bg, noise_seg], dim=1)
	else:
		noise = noise_bg
	out = [torch.unsqueeze(refine_net(input[:, i, ...] * mask[:, i:i + 1, ...] 
										+ noise * (1. - mask[:, i:i + 1, ...])
										   , flow[:, :, i, :, :]
									  ), 1) for i in range(opt.vid_length)]

	out = torch.cat(out, 1)
	return out

def make_color_wheel():
	"""
	Generate color wheel according Middlebury color code
	:return: Color wheel
	"""
	RY = 15
	YG = 6
	GC = 4
	CB = 11
	BM = 13
	MR = 6

	ncols = RY + YG + GC + CB + BM + MR

	colorwheel = np.zeros([ncols, 3])

	col = 0

	# RY
	colorwheel[0:RY, 0] = 255
	colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
	col += RY

	# YG
	colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
	colorwheel[col:col + YG, 1] = 255
	col += YG

	# GC
	colorwheel[col:col + GC, 1] = 255
	colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
	col += GC

	# CB
	colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
	colorwheel[col:col + CB, 2] = 255
	col += CB

	# BM
	colorwheel[col:col + BM, 2] = 255
	colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
	col += + BM

	# MR
	colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
	colorwheel[col:col + MR, 0] = 255

	return colorwheel



def compute_color(u, v):
	"""
	compute optical flow color map
	:param u: optical flow horizontal map
	:param v: optical flow vertical map
	:return: optical flow in color code
	"""
	[h, w] = u.shape
	img = np.zeros([h, w, 3])
	nanIdx = np.isnan(u) | np.isnan(v)
	u[nanIdx] = 0
	v[nanIdx] = 0

	colorwheel = make_color_wheel()
	ncols = np.size(colorwheel, 0)

	rad = np.sqrt(u ** 2 + v ** 2)

	a = np.arctan2(-v, -u) / np.pi

	fk = (a + 1) / 2 * (ncols - 1) + 1

	k0 = np.floor(fk).astype(int)

	k1 = k0 + 1
	k1[k1 == ncols + 1] = 1
	f = fk - k0

	for i in range(0, np.size(colorwheel, 1)):
		tmp = colorwheel[:, i]
		col0 = tmp[k0 - 1] / 255
		col1 = tmp[k1 - 1] / 255
		col = (1 - f) * col0 + f * col1

		idx = rad <= 1
		col[idx] = 1 - rad[idx] * (1 - col[idx])
		notidx = np.logical_not(idx)

		col[notidx] *= 0.75
		img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))

	return img

def gradientx(img):
	return img[:, :, :, :-1] - img[:, :, :, 1:]


def gradienty(img):
	return img[:, :, :-1, :] - img[:, :, 1:, :]



def attn_make_color_wheel():
    RY, YG, GC, CB, BM, MR = (15, 6, 4, 11, 13, 6)
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros([ncols, 3])
    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255
    return colorwheel

def attn_compute_color(u,v):
    h, w = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0
    # colorwheel = COLORWHEEL
    colorwheel = attn_make_color_wheel()
    ncols = np.size(colorwheel, 0)
    rad = np.sqrt(u**2+v**2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a+1) / 2 * (ncols - 1) + 1
    k0 = np.floor(fk).astype(int)
    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0
    for i in range(np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)
        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))
    return img



def flow_to_image(flow):
	"""Transfer flow map to image.
	Part of code forked from flownet.
	"""
	out = []
	maxu = -999.
	maxv = -999.
	minu = 999.
	minv = 999.
	maxrad = -1
	for i in range(flow.shape[0]):
		u = flow[i, :, :, 0]
		v = flow[i, :, :, 1]
		idxunknow = (abs(u) > 1e7) | (abs(v) > 1e7)
		u[idxunknow] = 0
		v[idxunknow] = 0
		maxu = max(maxu, np.max(u))
		minu = min(minu, np.min(u))
		maxv = max(maxv, np.max(v))
		minv = min(minv, np.min(v))
		rad = np.sqrt(u ** 2 + v ** 2)
		maxrad = max(maxrad, np.max(rad))
		u = u/(maxrad + np.finfo(float).eps)
		v = v/(maxrad + np.finfo(float).eps)
		img = attn_compute_color(u, v)
		out.append(img)
	return np.float32(np.uint8(out)) / 127.5 - 1.
