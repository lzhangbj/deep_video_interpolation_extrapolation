import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from nets.vgg import *
from utils.net_utils import *
from losses import *
from nets.SubNets import SegEncoder

MODE_LIST = ['s2s', 'x2x', 'xs2s', 'xs2x']

mean = torch.FloatTensor([0.485, 0.456, 0.406]).view([1,3,1,1])
std = torch.FloatTensor([0.229, 0.224, 0.225]).view([1,3,1,1])


class double_conv(nn.Module):
	'''(conv => BN => ReLU) * 2'''
	def __init__(self, in_ch, out_ch):
		super(double_conv, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, 3, padding=1),
			nn.BatchNorm2d(out_ch),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(out_ch, out_ch, 3, padding=1),
			nn.BatchNorm2d(out_ch),
			nn.LeakyReLU(0.2, inplace=True)
		)

	def forward(self, x):
		x = self.conv(x)
		return x


class inconv(nn.Module):
	'''
		equals to double_conv
	'''
	def __init__(self, in_ch, out_ch):
		super(inconv, self).__init__()
		self.conv = double_conv(in_ch, out_ch)

	def forward(self, x):
		x = self.conv(x)
		return x


class down(nn.Module):
	'''
		one encoder down process
	'''
	def __init__(self, in_ch, out_ch):
		super(down, self).__init__()
		self.mpconv = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
			nn.BatchNorm2d(out_ch),
			nn.LeakyReLU(0.2, inplace=True), 
			double_conv(out_ch, out_ch)
		)

	def forward(self, x):
		x = self.mpconv(x)
		return x


class up(nn.Module):
	def __init__(self, in_ch, out_ch, bilinear=True):
		super(up, self).__init__()

		#  would be a nice idea if the upsampling could be learned too,
		#  but my machine do not have enough memory to handle all those weights
		if bilinear:
			self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
		else:
			self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

		self.conv = double_conv(in_ch, out_ch)

	def forward(self, x1, x2=None):
		x1 = self.up(x1)
		if x2 is not None:
			# input is CHW
			diffY = x2.size()[2] - x1.size()[2]
			diffX = x2.size()[3] - x1.size()[3]

			x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
							diffY // 2, diffY - diffY//2))
		
			# for padding issues, see 
			# https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
			# https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

			x = torch.cat([x2, x1], dim=1)
		else:
			x = x1
		x = self.conv(x)
		return x


class outconv(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(outconv, self).__init__()
		self.conv = nn.Conv2d(in_ch, out_ch, 1)

	def forward(self, x):
		x = self.conv(x)
		return x


class UNet(nn.Module):
	def __init__(self, args):
		super(UNet, self).__init__()
		self.args=args
		self.in_channel = (3+4)*2


		self.seg_encoder = SegEncoder(in_dim=20)

		self.encoder_0 = inconv(self.in_channel, 64) 	# 64, 128, 256
		self.encoder_1 = down(64, 128) 					# 128, 64, 128
		self.encoder_2 = down(128, 256) 				# 256, 32, 64
		self.encoder_3 = down(256, 256)					# 256, 16, 32
		# ### dilated conv is to find the motion 
		# self.dil_conv = nn.Sequential(
		# 	nn.Conv2d(256, 256, 3, 1, 2, dilation=2),
		# 	nn.Conv2d(256, 256, 3, 1, 4, dilation=4),
		# 	nn.Conv2d(256, 256, 3, 1, 8, dilation=8)
		# 	)
		self.decoder_3 = up(256, 256) 				# 256, 32, 64
		self.decoder_2 = up(512, 128) 					# 128, 64, 128 
		self.decoder_1 = up(256, 64) 					# 64,  128, 256
		self.decoder_0 = inconv(128, 32)				# 32,  128, 256 

		self.rgb_decoder = nn.Conv2d(32, 3, 3, padding=1)

		self.seg_decoder = nn.Conv2d(32, 20, 3, padding=1)



	def forward(self, input, fg_mask=None, gt=None):

		seg_encoded = torch.cat([self.seg_encoder(input[:, 6+ i*20: 6 + (i+1)*20]) for i in range(2)], dim=1)
		input_ = torch.cat([input[:, :6], seg_encoded], dim=1)

		encon0 = self.encoder_0(input_)
		encon1 = self.encoder_1(encon0)
		encon2 = self.encoder_2(encon1)
		encon3 = self.encoder_3(encon2)

		decon3 = self.decoder_3(encon3)
		decon2 = self.decoder_2(torch.cat([decon3, encon2], dim=1))
		decon1 = self.decoder_1(torch.cat([decon2, encon1], dim=1))
		decon0 = self.decoder_0(torch.cat([decon1, encon0], dim=1))

		output_rgb = F.tanh(self.rgb_decoder(decon0))
		output_seg = self.seg_decoder(decon0) 

		return output_rgb, output_seg




# class UNet2SkipStream(nn.Module):
# 	def __init__(self, n_classes, seg_id=False,args=None):
# 		super(UNet2SkipStream, self).__init__()
# 		self.seg_id = seg_id
# 		self.n_classes = n_classes
# 		self.args=args
# 		if not seg_id:
# 			self.in_channel = (3+n_classes)*2
# 		else:
# 			self.in_channel = (3+1)*2

# 		# encoder
# 		self.encoder_inconv = inconv(self.in_channel, 64)
# 		self.encoder_down1 = down(64,128)
# 		self.encoder_down2 = down(128, 256)


# 		### dilated conv is to find the motion 
# 		self.dil_conv = nn.Sequential(
# 			nn.Conv2d(256, 256, 3, 1, 2, dilation=2),
# 			nn.Conv2d(256, 256, 3, 1, 4, dilation=4),
# 			nn.Conv2d(256, 256, 3, 1, 8, dilation=8)
# 			)

# 		# rgb decoder
# 		self.rgb_inconv = inconv(256, 256)
# 		self.rgb_up1 = up(256+128, 128)
# 		self.rgb_up2 = up(128+64, 64)
# 		self.rgb_outconv = outconv(64, 3)

# 		# seg decoder
# 		self.seg_inconv = inconv(256, 256)
# 		self.seg_up1 = up(256+128, 128)
# 		self.seg_up2 = up(128+64, 64)
# 		self.seg_outconv = outconv(64, self.n_classes)

# 		self.RGBLoss = RGBLoss(args)
# 		self.SegLoss = nn.CrossEntropyLoss()


# 	def forward(self, input, gt=None):
# 		# go through encoder
# 		encoder_1 = self.encoder_inconv(input)
# 		encoder_2 = self.encoder_down1(encoder_1)
# 		encoder_3 = self.encoder_down2(encoder_2)

# 		# go through dilated layer
# 		dil_out = self.dil_conv(encoder_3)

# 		# rgb encoder
# 		# rgb_1 = self.rgb_inconv(dil_out)
# 		# rgb_2 = self.rgb_up1(rgb_1, encoder_2)
# 		# rgb_3 = self.rgb_up2(rgb_2, encoder_1)
# 		# output_rgb = F.tanh(self.rgb_outconv(rgb_3))

# 		output_rgb = F.tanh(
# 			self.rgb_outconv(
# 				self.rgb_up2(
# 					self.rgb_up1(
# 						self.rgb_inconv(dil_out),
# 						encoder_2), 
# 					encoder_1)
# 				)
# 			)

# 		# seg encoder
# 		# seg_1 = self.seg_inconv(dil_out)
# 		# seg_2 = self.seg_up1(seg_1, encoder_2)
# 		# seg_3 = self.seg_up2(seg_2, encoder_1)
# 		# output_seg = self.seg_outconv(seg_3)

# 		output_seg = self.seg_outconv(
# 			self.seg_up2(
# 				self.seg_up1(
# 					self.seg_inconv(dil_out), 
# 					encoder_2), 
# 				encoder_1)
# 			)

# 		if self.training:
# 			l1_loss, gdl_loss, vgg_loss, ssim_loss = self.RGBLoss(output_rgb, gt[:, :3*1], normed=False, length=1)
# 			ce_loss = 0
# 			for i in range(1):
# 				ce_loss += self.args.ce_weight*self.SegLoss( output_seg[:,self.n_classes*i:self.n_classes*(i+1)], \
# 													torch.argmax(gt[:,3*1+self.n_classes*i:3*1+self.n_classes*(i+1)], dim=1).cuda(gt.get_device()))
# 			ce_loss/=1
# 			return output_rgb, output_seg, l1_loss, gdl_loss, vgg_loss, ssim_loss, ce_loss
# 		else:
# 			return output_rgb, output_seg

# 		# print(output_rgb.size())

# 		# return output_rgb, output_seg


	# def vis_feat(self, input):
	# 	# go through encoder
	# 	encoder_1 = self.encoder_inconv(input)
	# 	encoder_2 = self.encoder_down1(encoder_1)
	# 	encoder_3 = self.encoder_down2(encoder_2)

	# 	# go through dilated layer
	# 	dil_out = self.dil_conv(encoder_3)

	# 	# rgb encoder
	# 	rgb_1 = self.rgb_inconv(dil_out)
	# 	rgb_2 = self.rgb_up1(rgb_1, encoder_2)
	# 	rgb_3 = self.rgb_up2(rgb_2, encoder_1)
	# 	rgb_out = self.rgb_outconv(rgb_3)
	# 	# output_rgb = F.tanh(rgb_out)

	# 	# seg encoder
	# 	seg_1 = self.seg_inconv(dil_out)
	# 	seg_2 = self.seg_up1(seg_1, encoder_2)
	# 	seg_3 = self.seg_up2(seg_2, encoder_1)
	# 	# output_seg = self.seg_outconv(seg_3)

	# 	feat_dict = {
	# 	'encoder_1':encoder_1[0],
	# 	'encoder_2':encoder_2[0],
	# 	'encoder_3':encoder_3[0],
	# 	'dil_out':dil_out[0],
	# 	'rgb_1':rgb_1[0],
	# 	'rgb_2':rgb_2[0],
	# 	'rgb_3':rgb_3[0],
	# 	'seg_1':seg_1[0],
	# 	'seg_2':seg_2[0],
	# 	'seg_3':seg_3[0]
	# 	}

	# 	return feat_dict

	# def vis_conv(self):
	# 	return 




# class UNet2SkipStreamVid(nn.Module):
# 	def __init__(self, n_classes, seg_id=False, length=8, args=None):
# 		super(UNet2SkipStreamVid, self).__init__()
# 		self.seg_id = seg_id
# 		self.n_classes = n_classes
# 		self.length=length
# 		self.args=args
# 		if not seg_id:
# 			self.in_channel = (3+n_classes)*2
# 		else:
# 			self.in_channel = (3+1)*2

# 		# encoder
# 		self.encoder_inconv = inconv(self.in_channel, 64)
# 		self.encoder_down1 = down(64,128)
# 		self.encoder_down2 = down(128, 256)


# 		### dilated conv is to find the motion 
# 		self.dil_conv = nn.Sequential(
# 			nn.Conv2d(256, 256, 3, 1, 2, dilation=2),
# 			nn.Conv2d(256, 256, 3, 1, 4, dilation=4),
# 			nn.Conv2d(256, 256, 3, 1, 8, dilation=8)
# 			)

# 		# rgb decoder
# 		self.rgb_inconv = inconv(256, 256)
# 		self.rgb_up1 = up(256+128, 256)
# 		self.rgb_up2 = up(256+64, 256)
# 		self.rgb_outconv = nn.Sequential(
# 			inconv(256, 128),
# 			outconv(128, 3*length)
# 			)

# 		# seg decoder
# 		self.seg_inconv = inconv(256, 512)
# 		self.seg_up1 = up(512+128, 256)
# 		self.seg_up2 = up(256+64, 256)
# 		self.seg_outconv = outconv(256, length*self.n_classes)

# 		self.RGBLoss = RGBLoss(args)
# 		self.SegLoss = nn.CrossEntropyLoss()

# 	def forward(self, input, gt=None):
# 		# go through encoder
# 		encoder_1 = self.encoder_inconv(input)
# 		encoder_2 = self.encoder_down1(encoder_1)
# 		encoder_3 = self.encoder_down2(encoder_2)
# 		# print(encoder_2.size())
# 		# print(encoder_3.size())

# 		# go through dilated layer
# 		dil_out = self.dil_conv(encoder_3)

# 		# rgb encoder
# 		rgb_1 = self.rgb_inconv(dil_out)
# 		rgb_2 = self.rgb_up1(rgb_1, encoder_2)
# 		rgb_3 = self.rgb_up2(rgb_2, encoder_1)
# 		output_rgb = F.tanh(self.rgb_outconv(rgb_3))


# 		# seg encoder
# 		seg_1 = self.seg_inconv(dil_out)
# 		seg_2 = self.seg_up1(seg_1, encoder_2)
# 		seg_3 = self.seg_up2(seg_2, encoder_1)
# 		output_seg = self.seg_outconv(seg_3)

# 		if self.training:
# 			l1_loss, gdl_loss, vgg_loss, ssim_loss = self.RGBLoss(output_rgb, gt[:, :3*self.length], normed=False, length=self.length)
# 			ce_loss = 0
# 			for i in range(self.length):
# 				ce_loss += self.args.ce_weight*self.SegLoss( output_seg[:,self.n_classes*i:self.n_classes*(i+1)], \
# 													torch.argmax(gt[:,3*self.length+self.n_classes*i:3*self.length+self.n_classes*(i+1)], dim=1).cuda(gt.get_device()))
			
# 			ce_loss/=self.length
# 			return output_rgb, output_seg, l1_loss, gdl_loss, vgg_loss, ssim_loss, ce_loss
# 		else:
# 			return output_rgb, output_seg
