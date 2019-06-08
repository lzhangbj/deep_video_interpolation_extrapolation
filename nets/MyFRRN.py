import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
# from nets.vgg import *
# from utils.net_utils import *
# from losses import RGBLoss

MODE_LIST = ['s2s', 'x2x', 'xs2s', 'xs2x']

class FRRU(nn.Module):
	def __init__(self, in_z_channel, in_y_channel, scale_ratio, botneck=False):
		super(FRRU, self).__init__()
		self.in_z_channel = in_z_channel
		self.in_y_channel = in_y_channel
		self.scale_ratio = scale_ratio
		# decrease size while increasing channel to avoid infomation loss
		self.down = nn.Conv2d(in_z_channel, in_y_channel, 1+scale_ratio, stride=scale_ratio, padding=(1+scale_ratio)//2)
		if not botneck:
			self.convs = nn.Sequential(
				nn.Conv2d(in_y_channel*2, in_y_channel, 3, stride=1, padding=1),
				nn.ELU(),
				nn.Conv2d(in_y_channel, in_y_channel, 3, stride=1, padding=1),
				nn.ELU()
				)
		else:
			self.convs = nn.Sequential(
				nn.Conv2d(in_y_channel*2, in_y_channel, 3, stride=1, padding=1),
				nn.ELU(),
				nn.Conv2d(in_y_channel, in_y_channel, 3, 1, 2, dilation=2),
				nn.ELU(),
				nn.Conv2d(in_y_channel, in_y_channel, 3, 1, 4, dilation=4),
				nn.ELU(),
				nn.Conv2d(in_y_channel, in_y_channel, 3, 1, 8, dilation=8),
				nn.ELU(),
				nn.Conv2d(in_y_channel, in_y_channel, 3, stride=1, padding=1),
				nn.ELU()
				)
		self.transform = nn.Sequential(
			nn.Upsample(scale_factor=scale_ratio, mode='bilinear', align_corners=True),
			nn.Conv2d(in_y_channel, in_z_channel, 3, stride=1, padding=1),
			nn.ELU(),
			nn.Conv2d(in_z_channel, in_z_channel, 3, stride=1, padding=1)
			)

	def forward(self, z, y):
		down_z = self.down(z)
		out_y = self.convs(torch.cat([down_z, y], dim=1))
		out_z = z + self.transform(out_y)
		return out_z, out_y

class Block(nn.Module):
	def __init__(self, in_channel, out_channel):
		super(Block, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_channel, 64, 3, 1, 1),
			nn.ELU(),
			nn.Conv2d(64, 128, 3, 1, 1),
			nn.ELU(),
			nn.Conv2d(128, 128, 3, 1, 1),
			nn.ELU(),
			nn.Conv2d(128, out_channel, 3, 1, 1)
			)
		self.shortcut_conv = nn.Conv2d(in_channel, out_channel, 3, 1, 1)

	def forward(self, input):
		return self.conv(input) + self.shortcut_conv(input)

class RGBTailBlock(nn.Module):
	def __init__(self, in_channel, out_channel=3):
		super(RGBTailBlock, self).__init__()
		self.conv = nn.Sequential(
			nn.ELU(),
			nn.Conv2d(in_channel, in_channel, 3, 1, 1),
			nn.ELU(),
			nn.Conv2d(in_channel, out_channel, 3, 1, 1)
			)
		self.shortcut_conv = nn.Conv2d(in_channel, out_channel, 3, 1, 1)

	def forward(self, input):
		return self.conv(input) + self.shortcut_conv(input)

class SegTailBlock(nn.Module):
	def __init__(self, in_channel, inter_channel):
		super(SegTailBlock, self).__init__()
		self.conv = nn.Sequential(
			nn.ELU(),
			nn.Conv2d(in_channel, inter_channel, 3, 1, 1),
			nn.ELU(),
			nn.Conv2d(inter_channel, 20, 3, 1, 1)
			)

	def forward(self, input):
		return self.conv(input)


class DownBlock(nn.Module):
	def __init__(self, in_channel, out_channel, act=True):
		super(DownBlock, self).__init__()
		if not act:
			self.conv = nn.Sequential(
				nn.Conv2d(in_channel, out_channel, 3, 2, 1),
				nn.ELU()
				)
		else:
			self.conv = nn.Sequential(
				nn.ELU(),
				nn.Conv2d(in_channel, out_channel, 3, 2, 1),
				nn.ELU()
				)

	def forward(self, input):
		return self.conv(input)

class UpBlock(nn.Module):
	def __init__(self, in_channel, out_channel):
		super(UpBlock, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_channel, out_channel, 3, 1, 1),
			nn.ELU()
			)

	def forward(self, input):
		return self.conv(
			F.interpolate(input, scale_factor=2, mode='bilinear', align_corners=True)
			)

class MyFRRN(nn.Module):
	def __init__(self, args=None):
		super(MyFRRN, self).__init__()
		self.n_classes = 20 
		self.seg_encode_dim = 4
		self.args=args
		if self.args.mode == 'xs2xs':
			self.in_channel = (3+self.seg_encode_dim)*2
		elif self.args.mode =='xss2x':
			self.in_channel=(3+self.seg_encode_dim)*2+self.seg_encode_dim
		elif self.args.mode == 'edge':
			self.in_channel = (3+self.seg_encode_dim)*2 + 1
		self.n_channels = [32,64,96]

		self.seg_encoder = nn.Sequential(
			nn.Conv2d(self.n_classes, 32, 3, 1, 1),
			nn.ELU(),
			nn.Conv2d(32, 32, 3,1,1),
			nn.ELU(),
			nn.Conv2d(32, self.seg_encode_dim, 3, 1, 1) 
			)

		self.head_conv = Block(self.in_channel, self.n_channels[0]) # 128 originally

		self.down_2 = DownBlock(self.n_channels[0], self.n_channels[1], act=True)
		self.FRRU_d2 = FRRU(self.n_channels[0], self.n_channels[1], 2)

		self.down_4 = DownBlock(self.n_channels[1], self.n_channels[2], act=False)
		self.FRRU_bottle_neck = FRRU(self.n_channels[0], self.n_channels[2], 4, botneck=True)

		self.up_4 = UpBlock(self.n_channels[2], self.n_channels[1]) # up has no act
		self.FRRU_u2 = FRRU(self.n_channels[0], self.n_channels[1], 2)

		self.rgb_stream = RGBTailBlock(self.n_channels[0], 3)

		if args.mode == 'xs2xs' or args.mode == 'edge':
			self.seg_stream = SegTailBlock(self.n_channels[0], 64)

		# self.RGBLoss = RGBLoss(self.args)
		# self.SegLoss = nn.CrossEntropyLoss()


	def forward(self, input, gt=None):
		if self.args.mode in ['xs2xs', 'edge']:
			encoded_segs = [
				self.seg_encoder(input[:, 6:6+self.n_classes]),
				self.seg_encoder(input[:, 6+self.n_classes:6+2*self.n_classes])
			]
		else:
			encoded_segs = [
				self.seg_encoder(input[:, 6:6+self.n_classes]),
				self.seg_encoder(input[:, 6+self.n_classes:6+2*self.n_classes]),
				self.seg_encoder(input[:, 6+2*self.n_classes:6+3*self.n_classes])
			]
		if self.args.mode == 'xs2xs':
			encoded_feat = torch.cat([input[:,:6]] + encoded_segs, dim=1)
		elif self.args.mode == 'edge':
			# print(input.size())
			encoded_feat = torch.cat([input[:,:6], input[:, 46:]] + encoded_segs, dim=1)


		z0 = self.head_conv(encoded_feat)


		y0 = self.down_2(z0)
		z1, y1 = self.FRRU_d2(z0, y0)

		y1 = self.down_4(y1)
		z2, y2 = self.FRRU_bottle_neck(z1, y1)

		y2 = self.up_4(y2)
		z3, y3 = self.FRRU_u2(z2, y2)

		rgb_stream_out = self.rgb_stream(z3)

		output_rgb = F.tanh(rgb_stream_out[:, :3])
		# output_prob = F.sigmoid(rgb_stream_out[:,3:])
		output_seg = None
		if self.args.mode in ['xs2xs','edge']:
			output_seg = self.seg_stream(z3)
		if self.args.runner == 'gen':
			return output_rgb, output_seg
		else:
			return output_rgb,  output_seg, None





