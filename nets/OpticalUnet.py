import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import numpy as np

def meshgrid(height, width):
	x_t = torch.matmul(
		torch.ones(height, 1), torch.linspace(-1.0, 1.0, width).view(1, width))
	y_t = torch.matmul(
		torch.linspace(-1.0, 1.0, height).view(height, 1), torch.ones(1, width))

	grid_x = x_t.view(1, height, width)
	grid_y = y_t.view(1, height, width)
	return grid_x, grid_y


class OpticalRefineNet(nn.Module):
	def __init__(self, args):
		super(OpticalRefineNet, self).__init__()
		self.args = args
		self.coarse_model = nets.__dict__[args.coarse_model](args)
		self.refine_model = nets.__dict__[args.refine_model](args)

	def forward(self, input, seg=None, gt_x=None, gt_seg=None): # remove fg_mask and gt
		for_output, for_flow, for_mask, back_output, back_flow, back_mask = self.coarse_model(input) #.detach()
		output = self.refine_model(for_output, for_mask, back_output, back_mask)
		return output, for_output, for_flow, for_mask, back_output, back_flow, back_mask


class OpticalUnet(nn.Module):
	def __init__(self, args):
		super(OpticalUnet, self).__init__()
		self.encoder_1 = nn.Sequential(                                         # 128x128       
				nn.Conv2d(6, 32, 7, 1, 7//2),  nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(32, 32, 7, 1, 7//2), nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(32, 32, 7, 1, 7//2), nn.LeakyReLU(0.2, inplace=True),
			) 
		self.encoder_2 = nn.Sequential(                                         
				nn.Conv2d(32, 64, 5, 2, 5//2), nn.LeakyReLU(0.2, inplace=True), # 64x64
				nn.Conv2d(64, 64, 5, 1, 5//2), nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(64, 64, 5, 1, 5//2), nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(64, 64, 5, 1, 5//2), nn.LeakyReLU(0.2, inplace=True)
			)
		self.encoder_3 = nn.Sequential(
				nn.Conv2d(64, 128, 3, 2, 3//2), nn.LeakyReLU(0.2, inplace=True), # 32x32
				nn.Conv2d(128, 128,  3, 1, 3//2), nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(128, 128,  3, 1, 3//2), nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(128, 128,  3, 1, 3//2), nn.LeakyReLU(0.2, inplace=True)
			)
		self.encoder_4 = nn.Sequential(
				nn.Conv2d(128, 256, 3, 2, 3//2), nn.LeakyReLU(0.2, inplace=True),   # 16x16
				nn.Conv2d(256, 256, 3, 1, 3//2), nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(256, 256, 3, 1, 3//2), nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(256, 256, 3, 1, 3//2), nn.LeakyReLU(0.2, inplace=True)
			)


		self.bottom_layer = nn.Sequential(
				nn.Conv2d(256, 512, 3, 2, 3//2), nn.LeakyReLU(0.2, inplace=True),   # 8x8
				nn.Conv2d(512, 512, 3, 1, 3//2), nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(512, 512, 3, 1, 3//2), nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(512, 512, 3, 1, 3//2), nn.LeakyReLU(0.2, inplace=True)
			)


		self.up_4 = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True))
		self.decoder_4 = nn.Sequential(
				nn.Conv2d(256*2, 256, 3, 1, 3//2), nn.LeakyReLU(0.2, inplace=True), # 16x16
				nn.Conv2d(256, 256, 3, 1, 3//2), nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(256, 256, 3, 1, 3//2), nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(256, 256, 3, 1, 3//2), nn.LeakyReLU(0.2, inplace=True)
			)
		self.up_3 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True))
		self.decoder_3 = nn.Sequential(
				nn.Conv2d(128*2, 128, 3, 1, 3//2), nn.LeakyReLU(0.2, inplace=True), # 16x16
				nn.Conv2d(128, 128, 3, 1, 3//2), nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(128, 128, 3, 1, 3//2), nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(128, 128, 3, 1, 3//2), nn.LeakyReLU(0.2, inplace=True)
			)
		self.up_2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True))
		self.decoder_2 = nn.Sequential(
				nn.Conv2d(64*2, 64, 3, 1, 3//2), nn.LeakyReLU(0.2, inplace=True),   # 16x16
				nn.Conv2d(64, 64, 3, 1, 3//2), nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(64, 64, 3, 1, 3//2), nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(64, 64, 3, 1, 3//2), nn.LeakyReLU(0.2, inplace=True)
			)
		self.up_1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True))
		self.decoder_1 = nn.Sequential(
				nn.Conv2d(32*2, 32, 3, 1, 3//2), nn.LeakyReLU(0.2, inplace=True),   # 16x16
				nn.Conv2d(32, 32, 3, 1, 3//2), nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(32, 32, 3, 1, 3//2), nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(32, 32, 3, 1, 3//2), nn.LeakyReLU(0.2, inplace=True)
			)

		self.output_layer = nn.Conv2d(32, 6, 5, 1, 5//2)

	def forward(self, input):
		H, W = input.size()[2:]
		x1 = self.encoder_1(input[:6])
		x2 = self.encoder_2(x1)
		x3 = self.encoder_3(x2)
		x4 = self.encoder_4(x3)

		out = self.bottom_layer(x4)
		
		out = self.up_4(out)
		out = self.decoder_4(torch.cat([out, x4], dim=1))
		out = self.up_3(out)
		out = self.decoder_3(torch.cat([out, x3], dim=1))
		out = self.up_2(out)
		out = self.decoder_2(torch.cat([out, x2], dim=1))
		out = self.up_1(out)
		out = self.decoder_1(torch.cat([out, x1], dim=1))

		out = F.tanh(self.output_layer(out)) 
		for_flow = out[:,:2]
		for_mask = out[:,2:3]
		back_flow = out[:,3:5]
		back_mask = out[:,5:6]
		# flow should be -1-1, but to increate the flow range, we directly use it without *0.5 like voxel flow

		grid_x, grid_y = meshgrid(H, W)
		grid_x = grid_x.repeat([input.size()[0], 1, 1]).cuda()
		grid_y = grid_y.repeat([input.size()[0], 1, 1]).cuda()

		for_coor_x  = grid_x - for_flow[:, 0]
		for_coor_y  = grid_y - for_flow[:, 1]
		back_coor_x = grid_x + back_flow[:, 0]
		back_coor_y = grid_y + back_flow[:, 1]

		for_output = F.grid_sample(
			input[:, 0:3],
			torch.stack([for_coor_x, for_coor_y], dim=3),
			padding_mode='border')
		back_output = F.grid_sample(
			input[:, 3:6, :, :],
			torch.stack([coor_x_2, coor_y_2], dim=3),
			padding_mode='border')

		for_mask = 0.5 * (1.0 + for_mask)
		for_mask = for_mask.repeat([1, 3, 1, 1])
		back_mask = 0.5 * (1.0 + back_mask)
		back_mask = back_mask.repeat([1, 3, 1, 1])
		for_output = for_mask * for_output
		back_output = back_mask * back_output

		return for_output, for_flow, for_mask, back_output, back_flow, back_mask


class RefineUnet(nn.Module):
	def __init__(self, args):
		super(RefineUnet, self).__init__()
		self.encoder_1 = nn.Sequential(                                         # 128x128       
				nn.Conv2d(8, 32, 3, 1, 3//2),  nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(32, 32, 3, 1, 3//2), nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(32, 32, 3, 1, 3//2), nn.LeakyReLU(0.2, inplace=True),
			) 
		self.encoder_2 = nn.Sequential(                                         
				nn.Conv2d(32, 64, 3, 2, 3//2), nn.LeakyReLU(0.2, inplace=True), # 64x64
				nn.Conv2d(64, 64, 3, 1, 3//2), nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(64, 64, 3, 1, 3//2), nn.LeakyReLU(0.2, inplace=True)
			)
		self.encoder_3 = nn.Sequential(
				nn.Conv2d(64, 128, 3, 2, 3//2), nn.LeakyReLU(0.2, inplace=True), # 32x32
				nn.Conv2d(128, 128,  3, 1, 3//2), nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(128, 128,  3, 1, 3//2), nn.LeakyReLU(0.2, inplace=True)
			)
		self.encoder_4 = nn.Sequential(
				nn.Conv2d(128, 256, 3, 2, 3//2), nn.LeakyReLU(0.2, inplace=True),   # 16x16
				nn.Conv2d(256, 256, 3, 1, 3//2), nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(256, 256, 3, 1, 3//2), nn.LeakyReLU(0.2, inplace=True)
			)


		self.bottom_layer = nn.Sequential(
				nn.Conv2d(256, 256, 3, 2, 3//2), nn.LeakyReLU(0.2, inplace=True),   # 8x8
				nn.Conv2d(256, 256, 3, 1, 3//2), nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(256, 256, 3, 1, 3//2), nn.LeakyReLU(0.2, inplace=True)
			)


		self.up_4 = nn.Sequential(nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True))
		self.decoder_4 = nn.Sequential(
				nn.Conv2d(256*2, 256, 3, 1, 3//2), nn.LeakyReLU(0.2, inplace=True), # 16x16
				nn.Conv2d(256, 256, 3, 1, 3//2), nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(256, 256, 3, 1, 3//2), nn.LeakyReLU(0.2, inplace=True)
			)
		self.up_3 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True))
		self.decoder_3 = nn.Sequential(
				nn.Conv2d(128*2, 128, 3, 1, 3//2), nn.LeakyReLU(0.2, inplace=True), # 16x16
				nn.Conv2d(128, 128, 3, 1, 3//2), nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(128, 128, 3, 1, 3//2), nn.LeakyReLU(0.2, inplace=True)
			)
		self.up_2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True))
		self.decoder_2 = nn.Sequential(
				nn.Conv2d(64*2, 64, 3, 1, 3//2), nn.LeakyReLU(0.2, inplace=True),   # 16x16
				nn.Conv2d(64, 64, 3, 1, 3//2), nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(64, 64, 3, 1, 3//2), nn.LeakyReLU(0.2, inplace=True)
			)
		self.up_1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True))
		self.decoder_1 = nn.Sequential(
				nn.Conv2d(32*2, 32, 3, 1, 3//2), nn.LeakyReLU(0.2, inplace=True),   # 16x16
				nn.Conv2d(32, 32, 3, 1, 3//2), nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(32, 32, 3, 1, 3//2), nn.LeakyReLU(0.2, inplace=True)
			)

		self.output_layer = nn.Conv2d(32, 3, 3, 1, 3//2)


	def forward(self, img1, mask1, img2, mask2):
		input = torch.cat([img1, mask1, img2, mask2], dim=1)

		x1 = self.encoder_1(input)
		x2 = self.encoder_2(x1)
		x3 = self.encoder_3(x2)
		x4 = self.encoder_4(x3)

		out = self.bottom_layer(x4)
		
		out = self.up_4(out)
		out = self.decoder_4(torch.cat([out, x4], dim=1))
		out = self.up_3(out)
		out = self.decoder_3(torch.cat([out, x3], dim=1))
		out = self.up_2(out)
		out = self.decoder_2(torch.cat([out, x2], dim=1))
		out = self.up_1(out)
		out = self.decoder_1(torch.cat([out, x1], dim=1))

		out = self.output_layer(out)

		return out  




