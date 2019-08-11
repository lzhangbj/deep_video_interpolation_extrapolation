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

class VideoDiscriminator(nn.Module):
	def __init__(self, args):
		super(VideoDiscriminator, self).__init__()
		self.args=args
		self.input_dim = 23 if self.args.seg_disc else 3
		self.layer = nn.Sequential(
				nn.Conv2d(3*self.input_dim, 32, 3, 1, 1),
				# nn.BatchNorm2d(64),
				nn.LeakyReLU(0.2,inplace=False),

				nn.Conv2d(32, 64, 5, 1, 2),
				nn.BatchNorm2d(64),
				nn.LeakyReLU(0.2,inplace=False),

				nn.Conv2d(64, 32, 3, 1, 1),
				nn.BatchNorm2d(32),
				nn.LeakyReLU(0.2,inplace=False),

				# downsize 1 32*64*64
				nn.Conv2d(32, 32, 3, 2, 1),
				nn.LeakyReLU(0.2, inplace=True),
				ResnetBlock(32, 32, 3),
				# ResnetBlock(32, 32, 3),


				# downsize 2 64*32*32
				nn.Conv2d(32, 64, 3, 2, 1),
				nn.LeakyReLU(0.2, inplace=True),
				ResnetBlock(64, 64, 3),
				# ResnetBlock(64, 64, 3),
				# downsize 3 128*16*16
				nn.Conv2d(64, 128, 3, 2, 1),
				nn.LeakyReLU(0.2, inplace=True),
				ResnetBlock(128, 128, 3),
				# ResnetBlock(128, 128, 3),
				# downsize 4 256*8*8
				nn.Conv2d(128, 256, 3, 2, 1),
				nn.LeakyReLU(0.2, inplace=True),
				# ResnetBlock(256, 256, 3),
				ResnetBlock(256, 256, 3),
				# out layer
				nn.Conv2d(256, 256, 3, 1, 1),
				# nn.Tanh(),
				nn.AvgPool2d(8)
			)

	def forward(self, x, seg, input_x, input_seg, bboxes=None):
		input = torch.cat([x, seg, input_x, input_seg], dim=1) if self.args.seg_disc else torch.cat([x, input_x], dim=1)
		output = self.layer(input)
		output = output.view(-1, 256).mean(dim=1)
		return output

class VideoLocalDiscriminator(nn.Module):
	def __init__(self, args):
		super(VideoLocalDiscriminator, self).__init__()
		self.args=args
		self.input_dim = 23 if self.args.seg_disc else 3
		self.layer = nn.Sequential(
				nn.Conv2d(3*self.input_dim, 64, 1, 1, 0),		# 1,1
				nn.LeakyReLU(0.2,inplace=False),
				nn.Conv2d(64, 64, 3, 1, 1),			# 1,3
				nn.BatchNorm2d(64),
				nn.LeakyReLU(0.2, inplace=True),
				# downsize 1 64*64*64
				nn.Conv2d(64, 64, 3, 2, 1),			# 2,7
				nn.BatchNorm2d(64),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(64, 64, 3, 1, 1),			# 2,11
				nn.BatchNorm2d(64),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(64, 64, 3, 1, 1),			# 2,15
				nn.BatchNorm2d(64),
				nn.LeakyReLU(0.2, inplace=True),
				# downsize 2 128*32*32
				nn.Conv2d(64, 128, 3, 2, 1),		# 4,19
				nn.BatchNorm2d(128),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(128, 128, 3, 1, 1),		# 4,27
				nn.BatchNorm2d(128),
				nn.LeakyReLU(0.2, inplace=True),
				# downsize 3 128*16*16
				nn.Conv2d(128, 128, 3, 2, 1),		# 8,35
				nn.BatchNorm2d(128),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(128, 128, 3, 1, 1),		# 8,51
				nn.BatchNorm2d(128),
				nn.LeakyReLU(0.2, inplace=True),
				# downsize 4 256*8*8
				nn.Conv2d(128, 256, 3, 2, 1),		# 16,69
				nn.BatchNorm2d(256),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(256, 256, 3, 1, 1),		# 16,101
				nn.BatchNorm2d(256),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(256, 64, 1, 1, 0),		# 16,101
				nn.BatchNorm2d(64),
				nn.LeakyReLU(0.2, inplace=True),
				# out layer
				nn.Conv2d(64, 1, 1, 1, 0)			# 16,101
			)

	def forward(self, x, seg, input_x, input_seg, bboxes=None):
		input = torch.cat([x, seg, input_x, input_seg], dim=1) if self.args.seg_disc else torch.cat([x, input_x], dim=1)
		output = self.layer(input)
		return output

class VideoSNDiscriminator(nn.Module):
	def __init__(self, args):
		super(VideoSNDiscriminator, self).__init__()
		self.args=args
		self.input_dim = 23 if self.args.seg_disc else 3
		self.layer = nn.Sequential(
				SpectralNorm(nn.Conv2d(3*self.input_dim, 32, 3, 1, 1)),
				nn.LeakyReLU(0.2,inplace=False),

				SpectralNorm(nn.Conv2d(32, 64, 5, 1, 2)),
				nn.LeakyReLU(0.2,inplace=False),
				SpectralNorm(nn.Conv2d(64, 32, 3, 1, 1)),
				nn.LeakyReLU(0.2,inplace=False),

				# downsize 1 32*64*64
				SpectralNorm(nn.Conv2d(32, 32, 3, 2, 1)),
				nn.LeakyReLU(0.2, inplace=True),
				ResnetSNBlock(32, 32, 3),
				# ResnetBlock(32, 32, 3),

				# downsize 2 64*32*32
				SpectralNorm(nn.Conv2d(32, 64, 3, 2, 1)),
				nn.LeakyReLU(0.2, inplace=True),
				ResnetSNBlock(64, 64, 3),
				# downsize 3 128*16*16
				SpectralNorm(nn.Conv2d(64, 128, 3, 2, 1)),
				nn.LeakyReLU(0.2, inplace=True),
				ResnetSNBlock(128, 128, 3),
				# downsize 4 256*8*8
				# SpectralNorm(nn.Conv2d(128, 256, 3, 2, 1)),
				# nn.LeakyReLU(0.2, inplace=True),
				# ResnetSNBlock(256, 256, 3),
				# out layer
				SpectralNorm(nn.Conv2d(128, 128, 3, 1, 1)),
				# SpectralNorm(nn.Conv2d(256, 256, 3, 1, 1)),
				# nn.Tanh(),
				nn.AvgPool2d(16)
			)

	def forward(self, x, seg, input_x, input_seg, bboxes=None):
		input = torch.cat([x, seg, input_x, input_seg], dim=1) if self.args.seg_disc else torch.cat([x, input_x], dim=1)
		output = self.layer(input)
		output = output.view(-1, 128).mean(dim=1)
		return output

class VideoSNLocalDiscriminator(nn.Module):
	def __init__(self, args):
		super(VideoSNLocalDiscriminator, self).__init__()
		self.args=args
		self.input_dim = 23 if self.args.seg_disc else 3
		self.layer = nn.Sequential(
				SpectralNorm(nn.Conv2d(3*self.input_dim, 64, 1, 1, 0)),		# 1,1
				nn.LeakyReLU(0.2,inplace=False),
				SpectralNorm(nn.Conv2d(64, 64, 3, 1, 1)),			# 1,3
				nn.LeakyReLU(0.2, inplace=True),
				# downsize 1 64*64*64
				SpectralNorm(nn.Conv2d(64, 64, 3, 2, 1)),			# 2,7
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(64, 64, 3, 1, 1)),			# 2,11
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(64, 64, 3, 1, 1)),			# 2,15
				nn.LeakyReLU(0.2, inplace=True),
				# downsize 2 128*32*32
				SpectralNorm(nn.Conv2d(64, 128, 3, 2, 1)),		# 4,19
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(128, 128, 3, 1, 1)),		# 4,27
				nn.LeakyReLU(0.2, inplace=True),
				# downsize 3 128*16*16
				SpectralNorm(nn.Conv2d(128, 128, 3, 2, 1)),		# 8,35
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(128, 128, 3, 1, 1)),		# 8,51
				nn.LeakyReLU(0.2, inplace=True),
				# downsize 4 256*8*8
				SpectralNorm(nn.Conv2d(128, 256, 3, 2, 1)),		# 16,69
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(256, 256, 3, 1, 1)),		# 16,101
				nn.LeakyReLU(0.2, inplace=True),
				SpectralNorm(nn.Conv2d(256, 64, 1, 1, 0)),		# 16,101
				nn.LeakyReLU(0.2, inplace=True),
				# out layer
				SpectralNorm(nn.Conv2d(64, 1, 1, 1, 0))			# 16,101
			)

	def forward(self, x, seg, input_x, input_seg, bboxes=None):
		input = torch.cat([x, seg, input_x, input_seg], dim=1) if self.args.seg_disc else torch.cat([x, input_x], dim=1)
		output = self.layer(input)
		return output

# class VideoPatchProposalNet(nn.Module):
# 	def __init__(self, args):
# 		super(VideoPatchProposalNet, self).__init__()
# 		self.args=args	
# 		self.input_dim = 23 if self.args.seg_disc else 3
# 		self.input_layer 		= nn.Sequential(
# 				nn.Conv2d(self.input_dim, 32, 3, 1, 1),
# 				nn.LeakyReLU(0.2, inplace=True),
# 				nn.Conv2d(32, 32, 3, 1, 1),
# 				nn.BatchNorm2d(32),
# 				nn.LeakyReLU(0.2, inplace=True),
# 				nn.Conv2d(32, 32, 3, 2, 1),				# 64*64
# 				nn.BatchNorm2d(32),
# 				nn.LeakyReLU(0.2, inplace=True),
# 				ResnetBlock(32, 32, 3),
# 				ResnetBlock(32, 32, 3)
# 			)
# 		# encoder layer
# 		setattr(self, 'encoder_layer_1', 
# 			nn.Sequential(
# 					nn.Conv2d(32, 32, 3, 1, 1),			# 32*32
# 					nn.BatchNorm2d(32),
# 					nn.LeakyReLU(0.2, inplace=True),
# 					ResnetBlock(32,32,3),
# 					ResnetBlock(32,32,3)
# 				)
# 			)
# 		setattr(self, 'encoder_layer_2', 
# 			nn.Sequential(
# 					nn.Conv2d(32, 64, 3, 2, 1),			# 16*16
# 					nn.BatchNorm2d(64),
# 					nn.LeakyReLU(0.2, inplace=True),
# 					ResnetBlock(64,64,3),
# 					ResnetBlock(64,64,3)
# 				)
# 			)
# 		setattr(self, 'encoder_layer_3',
# 			nn.Sequential(
# 					nn.Conv2d(64, 128, 3, 2, 1),
# 					nn.LeakyReLU(0.2, inplace=True),
# 					nn.Conv2d(128, 128, 3, 1, 1),
# 					nn.LeakyReLU(0.2, inplace=True),
# 					ResnetBlock(128,128,3),
# 					ResnetBlock(128,128,3)
# 				)
# 			)

# 	def forward(self, gt_x, gt_seg, input_x, input_seg):



# class VideoAttnDiscriminator(nn.Module):
# 	def __init__(self, args):
# 		super(VideoAttnDiscriminator, self).__init__()
# 		self.args=args
# 		self.input_dim = 23 if self.args.seg_disc else 3
# 		self.layer = nn.Sequential(
# 				nn.Conv2d(3*self.input_dim, 64, 1, 1, 0),		# 1,1
# 				nn.LeakyReLU(0.2,inplace=False),
# 				nn.Conv2d(64, 64, 3, 1, 1),			# 1,3
# 				nn.BatchNorm2d(64),
# 				nn.LeakyReLU(0.2, inplace=True),
# 				# downsize 1 64*64*64
# 				nn.Conv2d(64, 64, 3, 2, 1),			# 2,7
# 				nn.BatchNorm2d(64),
# 				nn.LeakyReLU(0.2, inplace=True),
# 				nn.Conv2d(64, 64, 3, 1, 1),			# 2,11
# 				nn.BatchNorm2d(64),
# 				nn.LeakyReLU(0.2, inplace=True),
# 				nn.Conv2d(64, 64, 3, 1, 1),			# 2,15
# 				nn.BatchNorm2d(64),
# 				nn.LeakyReLU(0.2, inplace=True),
# 				# downsize 2 128*32*32
# 				nn.Conv2d(64, 128, 3, 2, 1),		# 4,19
# 				nn.BatchNorm2d(128),
# 				nn.LeakyReLU(0.2, inplace=True),
# 				nn.Conv2d(128, 128, 3, 1, 1),		# 4,27
# 				nn.BatchNorm2d(128),
# 				nn.LeakyReLU(0.2, inplace=True),
# 				# downsize 3 128*16*16
# 				nn.Conv2d(128, 128, 3, 2, 1),		# 8,35
# 				nn.BatchNorm2d(128),
# 				nn.LeakyReLU(0.2, inplace=True),
# 				nn.Conv2d(128, 128, 3, 1, 1),		# 8,51
# 				nn.BatchNorm2d(128),
# 				nn.LeakyReLU(0.2, inplace=True),
# 				# downsize 4 256*8*8
# 				nn.Conv2d(128, 256, 3, 2, 1),		# 16,69
# 				nn.BatchNorm2d(256),
# 				nn.LeakyReLU(0.2, inplace=True),
# 				nn.Conv2d(256, 256, 3, 1, 1),		# 16,101
# 				nn.BatchNorm2d(256),
# 				nn.LeakyReLU(0.2, inplace=True),
# 				nn.Conv2d(256, 64, 1, 1, 0),		# 16,101
# 				nn.BatchNorm2d(64),
# 				nn.LeakyReLU(0.2, inplace=True),
# 				# out layer
# 				nn.Conv2d(64, 1, 1, 1, 0)			# 16,101
# 			)

# 	def forward(self, x, seg, input_x, input_seg):
# 		input = torch.cat([x, seg, input_x, input_seg], dim=1) if self.args.seg_disc else torch.cat([x, input_x], dim=1)
# 		output = self.layer(input)
# 		return output