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

	def forward(self, input, fg_mask=None, gt=None, seg=None, gt_x=None, gt_seg=None):
		# if not self.args.high_res:
		# 	coarse_rgb, output_seg = self.coarse_model(input)
		# 	if self.training:
		# 		refine_rgbs, _ = self.refine_model(coarse_rgb.detach(), gt[:, 3:23], input[:,:6])
		# 	else:
		# 		refine_rgbs, _ = self.refine_model(coarse_rgb.detach(), output_seg, input[:,:6])
			
		# 	return coarse_rgb, refine_rgbs, output_seg

		# else:
		input_x_scaled = F.interpolate(input, scale_factor=0.5, mode='bilinear', align_corners=True) if self.args.high_res else input
		# print('x',input_x_scaled.size())
		# print('seg',seg.size())
		low_input = torch.cat([input_x_scaled, seg], dim=1) if self.args.mode == 'xs2xs' else input_x_scaled
		coarse_rgb, output_seg = self.coarse_model(low_input)

		if self.training:
			refine_rgbs, low_feature, flow = self.refine_model(coarse_rgb, gt_seg, input_x_scaled) #.detach()
		else:
			refine_rgbs, low_feature, flow = self.refine_model(coarse_rgb, output_seg, input_x_scaled) #.detach()

		re_refine_rgb = None
		if self.args.re_ref:
			re_refine_rgb, flow = self.re_ref_model(refine_rgbs[-1].detach(), input_x_scaled)

		# high res generator
		high_res_rgb = None
		if self.args.high_res:
			high_res_rgb = self.high_res_model(refine_rgbs[-1], low_feature.detach(), input)

		return  coarse_rgb, refine_rgbs, high_res_rgb, re_refine_rgb, output_seg, flow

		

