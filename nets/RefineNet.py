import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from nets.vgg import *
from utils.net_utils import *
from losses import *
import nets
from nets.SubNets import SegEncoder
from nets.UNet import *
from nets.MyFRRN import MyFRRN

class RefineNet(nn.Module):
	def __init__(self, args):
		super(RefineNet, self).__init__()
		self.args = args
		self.coarse_model = nets.__dict__[args.coarse_model](args)
		self.refine_model = nets.__dict__[args.refine_model](args)
		if args.high_res:
			self.high_res_model = nets.__dict__[args.high_res_model](args)
		if args.re_ref:
			self.re_ref_model = nets.__dict__[args.re_ref_model](args)

	def forward(self, input, seg=None, gt_x=None, gt_seg=None): # remove fg_mask and gt
		low_input = torch.cat([input, seg], dim=1) if self.args.mode == 'xs2xs' else input

		if self.args.syn_type == 'extra':
			if self.args.inpaint:
				coarse_rgb, output_seg, mask, inpainted_rgb = self.coarse_model(low_input)
				return coarse_rgb, output_seg, mask, inpainted_rgb
			else:
				coarse_rgb, output_seg = self.coarse_model(low_input)
				return coarse_rgb, output_seg 
			
		else: # inter
			if self.training:
				if not self.args.lock_refine:
					refine_rgbs, low_feature = self.refine_model(coarse_rgb, gt_seg, input) #.detach()
			else:
				refine_rgbs, low_feature= self.refine_model(coarse_rgb, output_seg, input) #.detach()
			return coarse_rgb, output_seg

			

