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
from nets.multi_scale_discriminator import MultiscaleDiscriminator

class RefineGAN(nn.Module):
	def __init__(self, args):
		super(RefineGAN, self).__init__()
		self.args = args
		self.coarse_model = nets.__dict__[args.coarse_model](args)
		self.refine_model = nets.__dict__[args.refine_model](args)
		self.discriminator = MultiscaleDiscriminator(23, ndf=64, n_layers=7, norm_layer=nn.BatchNorm2d, \
			  use_sigmoid=True, num_D=2, getIntermFeat=False)

	def set_net_grad(self, net, flag=True):
		for p in net.parameters():
			p.requires_grad = flag

	def forward(self, input, fg_mask=None, gt=None):
		coarse_rgb, seg = self.coarse_model(input)
		refine_rgbs = self.refine_model(coarse_rgb.detach())

		
		# Fake Detection and Loss
		pred_fake_D = self.discriminator(torch.cat([refine_rgbs[-1].detach(), gt[:,3:23]],dim=1))  

		# Real Detection and Loss       
		pred_real_D = self.discriminator(gt)
		if not self.args.val:
			# GAN loss (Fake Possibility Loss)     
			self.set_net_grad(self.discriminator, False)   
			pred_fake_G = self.discriminator(torch.cat([refine_rgbs[-1], gt[:,3:23]], dim=1))
			self.set_net_grad(self.discriminator, True)

			return coarse_rgb, refine_rgbs, seg, pred_fake_D, pred_real_D, pred_fake_G
		else:
			return coarse_rgb, refine_rgbs, seg, pred_fake_D, pred_real_D

