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


class FrameDiscriminator(nn.Module):
	def __init__(self, args):
		super(FrameDiscriminator, self).__init__()
		self.args=args
		self.input_dim = 23 if self.args.seg_disc else 3
		self.layer = nn.Sequential(
				nn.Conv2d(self.input_dim, 16, 3, 1, 1),
				nn.LeakyReLU(0.2,inplace=False),

				nn.Conv2d(16, 32, 5, 1, 2),
				nn.BatchNorm2d(32),
				nn.LeakyReLU(0.2,inplace=False),

				# downsize 1 64*64*64
				nn.Conv2d(32, 64, 3, 2, 1),
				nn.LeakyReLU(0.2, inplace=True),
				ResnetBlock(64, 64, 3),
				# downsize 2 96*32*32
				nn.Conv2d(64, 96, 3, 2, 1),
				nn.LeakyReLU(0.2, inplace=True),
				ResnetBlock(96, 96, 3),
				# downsize 3 128*16*16
				nn.Conv2d(96, 128, 3, 2, 1),
				nn.LeakyReLU(0.2, inplace=True),
				ResnetBlock(128, 128, 3),
				# downsize 4 192*8*8
				nn.Conv2d(128, 192, 3, 2, 1),
				nn.LeakyReLU(0.2, inplace=True),
				ResnetBlock(192, 192, 3),
				# out layer
				nn.Conv2d(192, 192, 3, 1, 1),
				nn.AvgPool2d(8)
			)

	def forward(self, x, seg, bboxes=None):
		input = x
		if self.args.seg_disc:
			input = torch.cat([x, seg], dim=1)
		output = self.layer(input)
		output = output.view(-1, 192).mean(dim=1)
		return output

class FrameLocalDiscriminator(nn.Module):
	def __init__(self, args):
		super(FrameLocalDiscriminator, self).__init__()
		self.args=args
		self.input_dim = 23 if self.args.seg_disc else 3
		self.layer = nn.Sequential(
				nn.Conv2d(self.input_dim, 16, 3, 1, 1),			# 1,3
				nn.LeakyReLU(0.2,inplace=False),
				nn.Conv2d(16, 32, 5, 1, 2),						# 1,7
				nn.BatchNorm2d(32),
				nn.LeakyReLU(0.2,inplace=False),

				# downsize 1  64*64*64
				nn.Conv2d(32, 64, 3, 2, 1),			# 2,9
				nn.BatchNorm2d(64),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(64, 64, 3, 1, 1),			# 2,13
				nn.BatchNorm2d(64),
				nn.LeakyReLU(0.2, inplace=True),
				# downsize 2 128*32*32
				nn.Conv2d(64, 128, 3, 2, 1),		# 4,17
				nn.BatchNorm2d(128),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(128, 128, 3, 1, 1),		# 4,25
				nn.BatchNorm2d(128),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(128, 64, 3, 1, 1),		# 4,33
				nn.BatchNorm2d(64),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(64,  1, 1, 1, 0)			# 8,33
			)

	def forward(self, x, seg, bboxes=None):
		input = x
		if self.args.seg_disc:
			input = torch.cat([x, seg], dim=1)
		output = self.layer(input)
		return output

class FrameSNDiscriminator(nn.Module):
	def __init__(self, args):
		super(FrameSNDiscriminator, self).__init__()
		self.args=args
		self.input_dim = 23 if self.args.seg_disc else 3
		self.layer = nn.Sequential(
				SpectralNorm(nn.Conv2d(self.input_dim, 16, 3, 1, 1)),
				nn.LeakyReLU(0.2,inplace=False),

				SpectralNorm(nn.Conv2d(16, 32, 5, 1, 2)),
				nn.LeakyReLU(0.2,inplace=False),

				# downsize 1 64*64*64
				SpectralNorm(nn.Conv2d(32, 64, 3, 2, 1)),
				nn.LeakyReLU(0.2, inplace=True),
				ResnetSNBlock(64, 64, 3),
				# downsize 2 96*32*32
				SpectralNorm(nn.Conv2d(64, 96, 3, 2, 1)),
				nn.LeakyReLU(0.2, inplace=True),
				ResnetSNBlock(96, 96, 3),
				# downsize 3 128*16*16
				SpectralNorm(nn.Conv2d(96, 128, 3, 2, 1)),
				nn.LeakyReLU(0.2, inplace=True),
				ResnetSNBlock(128, 128, 3),
				# downsize 4 192*8*8
				SpectralNorm(nn.Conv2d(128, 192, 3, 2, 1)),
				nn.LeakyReLU(0.2, inplace=True),
				ResnetSNBlock(192, 192, 3),
				# out layer
				SpectralNorm(nn.Conv2d(192, 192, 3, 1, 1)),
				nn.AvgPool2d(8)
			)

	def forward(self, x, seg, bboxes=None):
		input = x
		if self.args.seg_disc:
			input = torch.cat([x, seg], dim=1)
		output = self.layer(input)
		output = output.view(-1, 192).mean(dim=1)
		return output

class FrameSNLocalDiscriminator(nn.Module):
	def __init__(self, args):
		super(FrameSNLocalDiscriminator, self).__init__()
		self.args=args
		self.input_dim = 23 if self.args.seg_disc else 3
		self.layer = nn.Sequential(
				SpectralNorm(nn.Conv2d(self.input_dim, 16, 3, 1, 1)),			# 1,3
				nn.LeakyReLU(0.2,inplace=False),
				SpectralNorm(nn.Conv2d(16, 32, 5, 1, 2)),						# 1,7
				nn.LeakyReLU(0.2,inplace=False),

				# downsize 1  64*64*64
				SpectralNorm(nn.Conv2d(32, 64, 3, 2, 1)),			# 2,9
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(64, 64, 3, 1, 1)),			# 2,13
				nn.LeakyReLU(0.2, inplace=True),
				# downsize 2 128*32*32
				SpectralNorm(nn.Conv2d(64, 128, 3, 2, 1)),		# 4,17
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(128, 128, 3, 1, 1)),		# 4,25
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(128, 64, 3, 1, 1)),		# 4,33
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(64,  1, 1, 1, 0))			# 8,33
			)

	def forward(self, x, seg, bboxes=None):
		input = x
		if self.args.seg_disc:
			input = torch.cat([x, seg], dim=1)
		output = self.layer(input)
		return output
