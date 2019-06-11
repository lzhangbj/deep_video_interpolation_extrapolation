import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from nets.vgg import *
from utils.net_utils import *
from losses import *
from nets.SubNets import SegEncoder
from nets.UNet import *


class SepUNet(nn.Module):
	def __init__(self, args):
		super(SepUNet, self).__init__()
		self.args=args
		self.in_channel = (3+4)*2


		self.seg_encoder = SegEncoder(in_dim=20)

		self.fg_encoder_0 = inconv(self.in_channel, 32) 	# 32, 128, 256
		self.fg_encoder_1 = down(32, 64) 					# 64, 64, 128
		self.fg_encoder_2 = down(64, 128) 					# 128, 32, 64
		self.fg_encoder_3 = down(128, 128)					# 128, 16, 32

		self.bg_encoder_0 = inconv(self.in_channel, 32) 	# 32, 128, 256
		self.bg_encoder_1 = down(32, 64) 					# 64, 64, 128
		self.bg_encoder_2 = down(64, 128) 					# 128, 32, 64
		self.bg_encoder_3 = down(128, 128)					# 128, 16, 32

		self.decoder_3 = up(256, 256) 					# 256, 32, 64
		self.decoder_2 = up(512, 128) 					# 128, 64, 128 
		self.decoder_1 = up(256, 64) 					# 64,  128, 256
		self.decoder_0 = inconv(128, 32)				# 32,  128, 256 

		self.rgb_decoder = nn.Conv2d(32, 3, 3, padding=1) # need to change

		self.seg_decoder = nn.Conv2d(32, 20, 3, padding=1)



	def forward(self, input, fg_mask=None, gt=None):
		seg_encoded = [self.seg_encoder(input[:, 6+ i*20: 6 + (i+1)*20]) for i in range(2)]
		# print('seg', seg_encoded[0].size())
		# print('mask', fg_mask.size())
		fg_seg_encoded = torch.cat([seg_encoded[i] * fg_mask[:,i:i+1] for i in range(2)], dim=1)
		bg_seg_encoded = torch.cat([seg_encoded[i] * (1-fg_mask[:,i:i+1]) for i in range(2)], dim=1)

		input_fg = torch.cat([input[:, :6], fg_seg_encoded], dim=1)
		input_bg = torch.cat([input[:, :6], bg_seg_encoded], dim=1)

		# fg
		fg_encon0 = self.fg_encoder_0(input_fg)
		fg_encon1 = self.fg_encoder_1(fg_encon0)
		fg_encon2 = self.fg_encoder_2(fg_encon1)
		fg_encon3 = self.fg_encoder_3(fg_encon2)
		# bg
		bg_encon0 = self.bg_encoder_0(input_bg)
		bg_encon1 = self.bg_encoder_1(bg_encon0)
		bg_encon2 = self.bg_encoder_2(bg_encon1)
		bg_encon3 = self.bg_encoder_3(bg_encon2)

		decon3 = self.decoder_3(torch.cat([fg_encon3, bg_encon3], dim=1))
		decon2 = self.decoder_2(torch.cat([decon3, fg_encon2, bg_encon2], dim=1))
		decon1 = self.decoder_1(torch.cat([decon2, fg_encon1, bg_encon1], dim=1))
		decon0 = self.decoder_0(torch.cat([decon1, fg_encon0, bg_encon0], dim=1))

		output_rgb = F.tanh(self.rgb_decoder(decon0))
		output_seg = self.seg_decoder(decon0) 

		return output_rgb, output_seg
