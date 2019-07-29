import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import nets


class InterNet(nn.Module):
	def __init__(self, args):
		super(InterNet, self).__init__()
		self.args = args
		self.coarse_model = nets.__dict__[args.coarse_model](args)

	def forward(self, input, seg=None): # remove fg_mask and gt
		low_input = torch.cat([input, seg], dim=1)
		coarse_rgb, coarse_seg = self.coarse_model(low_input)
		return coarse_rgb, coarse_seg 

		

