import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import nets


class ExtraInpaintNet(nn.Module):
	def __init__(self, args):
		super(ExtraInpaintNet, self).__init__()
		self.args = args
		self.coarse_model = nets.__dict__[args.coarse_model](args)
		self.inpaint_model = nets.__dict__[args.inpaint_model](args)

	def forward(self, input, seg=None, gt_x=None, gt_seg=None):
		low_input = torch.cat([input, seg], dim=1) 
		coarse_rgb, output_seg, mask = self.coarse_model(low_input)
		inpainted_rgb = self.inpaint_model(coarse_rgb, mask, output_seg)
		return coarse_rgb, output_seg, mask, inpainted_rgb
