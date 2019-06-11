import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from nets.vgg import my_vgg
from utils.net_utils import *
from losses import *

MODE_LIST = ['s2s', 'x2x', 'xs2s', 'xs2x']

mean = torch.FloatTensor([0.485, 0.456, 0.406]).view([1,3,1,1])
std = torch.FloatTensor([0.229, 0.224, 0.225]).view([1,3,1,1])

class SegEncoder(nn.Module):
	def __init__(self, in_dim, out_dim=4):
		super(SegEncoder, self).__init__()
		self.in_dim= in_dim
		self.out_dim = out_dim
		self.sequence = nn.Sequential(
			nn.Conv2d(in_dim, 32, 3, 1, 1),
			nn.BatchNorm2d(32),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(32, 32, 3,1,1),
			nn.BatchNorm2d(32),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(32, out_dim, 3, 1, 1) 
			)
	def forward(self, input):
		return self.sequence(input)

'''
	(bs, (3+segdim)*vid_len, 128, 128)
	sequence            (bs, 64, 8, 8)
	mu_fc       --> (bs*4, 512)
	logvar_fc   --> (bs*4, 512) 
'''
class FlowEncoder(nn.Module):
	def __init__(self, args, in_dim, latent_dim=512):
		super(FlowEncoder, self).__init__()
		self.args = args
		self.in_dim=in_dim
		self.sequence = nn.Sequential(
			nn.Conv2d(self.in_dim, 64, 5, 2, 2, bias=False), # 64 3*(args.vid_length+1)+args.seg_dim
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(64, 64, 5, 2, 2, bias=False),  # 32
			nn.BatchNorm2d(64),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(64, 64, 3, 1, 1, bias=False),  # 32
			nn.BatchNorm2d(64),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(64, 128, 5, 2, 1, bias=False),  # 16
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(128, 128, 3, 1, 1, bias=False),  # 16
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(128, 48, 5, 2, 2, bias=False)  # 8
			)
		self.mu_fc = nn.Linear(1024, latent_dim)
		self.logvar_fc = nn.Linear(1024, latent_dim)
	def forward(self, input):
		seq_out = self.sequence(input).view(-1, 1024)

		mu_vec = self.mu_fc(seq_out)
		logvar_vec = self.logvar_fc(seq_out)
		return mu_vec, logvar_vec


################################### blocks ###########################

'''
	3d conv + 3d bn
'''
class gateconv3d(nn.Module):
	def __init__(self, innum, outnum, kernel, stride, pad):
		super(gateconv3d, self).__init__()
		self.conv = nn.Conv3d(innum, outnum, kernel, stride, pad, bias=True)
		self.bn = nn.BatchNorm3d(outnum)

	def forward(self, x):
		return F.leaky_relu(self.bn(self.conv(x)), 0.2)

'''
	conv + bn + LeakyReLU
'''
class convblock(nn.Module):
	def __init__(self, innum, outnum, kernel, stride, pad):
		super(convblock, self).__init__()
		self.main = nn.Sequential(
			nn.Conv2d(innum, outnum, kernel, stride, pad, bias=False),
			nn.BatchNorm2d(outnum),
			nn.LeakyReLU(0.2, inplace=True))

	def forward(self, x):
		return self.main(x)


'''
	conv + LeakyReLU
'''
class convbase(nn.Module):
	def __init__(self, innum, outnum, kernel, stride, pad):
		super(convbase, self).__init__()
		self.main = nn.Sequential(
			nn.Conv2d(innum, outnum, kernel, stride, pad),
			nn.LeakyReLU(0.2, inplace=True))

	def forward(self, x):
		return self.main(x)


'''
	conv + bn + LeakyReLU 
	+ conv + bn + LeakyReLU 
	+ bi-upsample
'''
class upconv(nn.Module):
	def __init__(self, innum, outnum, kernel, stride, pad):
		super(upconv, self).__init__()
		self.main = nn.Sequential(
			nn.Conv2d(innum, outnum * 2, kernel, stride, pad),
			nn.BatchNorm2d(outnum * 2),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(outnum * 2, outnum, kernel, stride, pad),
			nn.BatchNorm2d(outnum),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Upsample(scale_factor=2, mode='bilinear')
		)

	def forward(self, x):
		return self.main(x)


'''
	conv + LeakyReLU            (32, 64, 64)
	+ conv + bn + LeakyReLU     (64, 32, 32)
	+ conv + bn + LeakyReLU     (128, 16, 16)
	+ conv + bn + LeakyReLU     (256, 8, 8)     

'''
class encoder(nn.Module):
	def __init__(self, args):
		super(encoder, self).__init__()
		self.econv1 = nn.Sequential(
				convbase(3+args.seg_dim, 32, 3,1,1),
				convblock(32, 32, 3,1,1)				# 32, 128, 128
			)

		self.econv2 = nn.Sequential(
				convblock(32, 64, 5, 2, 2),
				convblock(64, 64, 3,1,1),  
				convblock(64, 64, 3,1,1)  		# 64,64,64
			)

		self.econv3 = nn.Sequential(
				convblock(64, 128, 5, 2, 2),  
				convblock(128, 128, 3, 1, 1),  
				convblock(128, 128, 3, 1, 1)  # 128,32,32
			)

		self.econv4 = nn.Sequential(
				convblock(128, 256, 5, 2, 2), 
				convblock(256, 256, 3, 1, 1)  # 256,16,16
			)

		self.econv5 = nn.Sequential(
				convblock(256, 256, 5, 2, 2),
				convblock(256, 256, 3,1,1)  # 256,8,8
			)
		# self.econv5_1 = 

	def forward(self, x):
		enco1 = self.econv1(x)  # 32
		enco2 = self.econv2(enco1)  # 64
		enco3 = self.econv3(enco2)  # 128
		codex = self.econv4(enco3)  # 256
		return enco1, enco2, enco3, codex

'''
	skip connection decoder

	skip connect single image+seg features into flow decoder in EVERY frame

	enco1 (bs, 32, 64, 64)
	enco2 (bs, 64, 32, 32)
	enco3 (bs, 128, 16, 16)
	z     (bs*4, 256+16, 8, 8)

	conv + bn + LeakyReLU           (bs*4, 256, 8, 8)
	+ upconv                        (bs, 128, 4, 16, 16)
		+ enco3-skip                (bs, 256, 4, 16, 16)
			+ combine               (bs*4, 256, 16, 16)

	+ upconv                        (bs*4, 64, 32, 32)
		+ chunk                     (bs, 64, 4, 32, 32)
			+ gateconv3d            (bs, 64, 4, 32, 32)
				+ enco2-skip        (bs, 128, 4, 32, 32)
					+ combine       (bs*4, 128, 32, 32)

	+ upconv                        (bs*4, 32, 64, 64)
		+ gateconv3d                (bs, 32, 4, 64, 64)
			+ enco1-skip            (bs, 64, 4, 64, 64)
				+ combine           (bs*4, 64, 64, 64)
'''
class decoder(nn.Module):
	def __init__(self, args):
		super(decoder, self).__init__()
		self.args = args
		self.dconv1 = convblock(256 + 16, 256, 3, 1, 1)  # 256,8,8
		self.dconv2 = upconv(256, 128, 3, 1, 1)  # 128,16,16
		self.dconv3 = upconv(256, 64, 3, 1, 1)  # 64,32,32
		self.dconv4 = upconv(128, 32, 3, 1, 1)  # 32,64,64
		self.gateconv1 = gateconv3d(64, 64, 3, 1, 1)
		self.gateconv2 = gateconv3d(32, 32, 3, 1, 1)

	def forward(self, enco1, enco2, enco3, z):
		args = self.args
		deco1 = self.dconv1(z)  # .view(-1,256,4,4,4)# bs*4,256,8,8
		deco2 = torch.cat(torch.chunk(self.dconv2(deco1).unsqueeze(2), args.vid_length, 0), 2)  # bs*4,128,16,16
		deco2 = torch.cat(torch.unbind(torch.cat([deco2, torch.unsqueeze(enco3, 2).repeat(1, 1, args.vid_length, 1, 1)], 1), 2), 0)
		deco3 = torch.cat(self.dconv3(deco2).unsqueeze(2).chunk(args.vid_length, 0), 2)  # 128,32,32
		deco3 = self.gateconv1(deco3)
		deco3 = torch.cat(torch.unbind(torch.cat([deco3, torch.unsqueeze(enco2, 2).repeat(1, 1, args.vid_length, 1, 1)], 1), 2), 0)
		deco4 = torch.cat(self.dconv4(deco3).unsqueeze(2).chunk(args.vid_length, 0), 2)  # 32,4,64,64
		deco4 = self.gateconv2(deco4)
		deco4 = torch.cat(torch.unbind(torch.cat([deco4, torch.unsqueeze(enco1, 2).repeat(1, 1, args.vid_length, 1, 1)], 1), 2), 0)
		return deco4

################################# mini nets ############################
'''
	upconv + conv2d  
	64 --> 2
'''
class getflow(nn.Module):
	def __init__(self):
		super(getflow, self).__init__()
		self.main = nn.Sequential(
			upconv(64, 16, 5, 1, 2),
			nn.Conv2d(16, 2, 5, 1, 2),
		)

	def forward(self, x):
		return self.main(x)

'''
	upconv + conv3d + sigmoid
	64 --> 2 
'''
class get_occlusion_mask(nn.Module):
	def __init__(self):
		super(get_occlusion_mask, self).__init__()
		self.main = nn.Sequential(
			upconv(64, 16, 5, 1, 2),
			nn.Conv2d(16, 2, 5, 1, 2),
		)

	def forward(self, x):
		return torch.sigmoid(self.main(x))

'''
	upconv + conv2d + sigmoid
	64 --> 3 rgb
'''
class get_frames(nn.Module):
	def __init__(self):
		super(get_frames, self).__init__()
		self.main = nn.Sequential(
			upconv(64, 16, 5, 1, 2),
			nn.Conv2d(16, 3, 5, 1, 2)
		)

	def forward(self, x):
		return torch.sigmoid(self.main(x))


