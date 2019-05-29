import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from utils.data_utils import *


def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm2d') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)



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
