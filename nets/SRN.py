import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from nets.vgg import *
from nets.resnet101 import *
from utils.net_utils import *
from losses import *
from time import time
from nets.SubNets import SegEncoder
from nets.UNet import *


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

################### 

class encoder_layer2(nn.Module):
	def __init__(self, in_dim, out_dim, ks):
		super(encoder_layer2, self).__init__()
		self.conv = nn.Sequential(
				nn.Conv2d(in_dim, out_dim, ks, stride=2, padding=ks//2),
				nn.LeakyReLU(0.2, inplace=True),
				ResnetBlock(out_dim, out_dim, 5),
				ResnetBlock(out_dim, out_dim, 5),
				ResnetBlock(out_dim, out_dim, 5)
			)

	def forward(self, input):
		return self.conv(input)

class encoder_layer3(nn.Module):
	def __init__(self, in_dim, out_dim, ks):
		super(encoder_layer3, self).__init__()
		self.conv = nn.Sequential(
				nn.Conv2d(in_dim, out_dim, ks, stride=2, padding=ks//2),
				nn.LeakyReLU(0.2, inplace=True),
				ResnetBlock(out_dim, out_dim, 3),
				ResnetBlock(out_dim, out_dim, 3),
				ResnetBlock(out_dim, out_dim, 3)
			)

	def forward(self, input):
		return self.conv(input)

class encoder_layer4(nn.Module):
	def __init__(self, in_dim, out_dim, ks):
		super(encoder_layer4, self).__init__()
		self.conv = nn.Sequential(
				nn.Conv2d(in_dim, out_dim, ks, stride=2, padding=ks//2),
				nn.LeakyReLU(0.2, inplace=True),
				ResnetBlock(out_dim, out_dim, 3)
			)

	def forward(self, input):
		return self.conv(input)

class decoder_layer4(nn.Module):
	def __init__(self, in_dim, out_dim, ks):
		super(decoder_layer4, self).__init__()
		self.conv = nn.Sequential(
				ResnetBlock(in_dim, in_dim, 5),
				ResnetBlock(in_dim, in_dim, 5),
				ResnetBlock(in_dim, in_dim, 5),
				nn.ConvTranspose2d(in_dim, out_dim, ks, stride=2, padding=1),
				nn.LeakyReLU(0.2, inplace=True)
			)

	def forward(self, input):
		return self.conv(input)

class decoder_layer5(nn.Module):
	def __init__(self, in_dim, out_dim, ks):
		super(decoder_layer5, self).__init__()
		self.conv = nn.Sequential(
				ResnetBlock(in_dim, in_dim, 3),
				ResnetBlock(in_dim, in_dim, 3),
				ResnetBlock(in_dim, in_dim, 3),
				nn.ConvTranspose2d(in_dim, out_dim, ks, stride=2, padding=1),
				nn.LeakyReLU(0.2, inplace=True)
			)

	def forward(self, input):
		return self.conv(input)



class decoder_layer_out(nn.Module):
	def __init__(self, in_dim, out_dim, ks, get_feature=False):
		super(decoder_layer_out, self).__init__()
		self.get_feature = get_feature
		self.conv = nn.Sequential(
				ResnetBlock(in_dim, in_dim, 5),
				ResnetBlock(in_dim, in_dim, 5),
				ResnetBlock(in_dim, in_dim, 5),
				# nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(in_dim, out_dim, ks, 1, padding=ks//2)
			)

	def forward(self, input):
		if not self.get_feature:
			return self.conv(input)
		else:
			feature = self.conv[:-1](input)
			return self.conv[-1](feature), feature



###################
class SRN4(nn.Module):
	def __init__(self, args):
		super(SRN4, self).__init__()
		self.n_scales = args.n_scales
		self.args=args
		self.encoder_1 = nn.Sequential(
				nn.Conv2d(6, 32, 5, stride=1, padding=5//2),
				nn.LeakyReLU(0.2, inplace=True),
				ResnetBlock(32, 32, 5),
				ResnetBlock(32, 32, 5),
				ResnetBlock(32, 32, 5)          
			)
		self.encoder_2 = encoder_layer2(32, 64, 5)
		self.encoder_3 = encoder_layer2(64, 128, 5)

		self.hidden_comb = nn.Sequential(
				nn.Conv2d(256, 128, 3, 1, 1),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(128, 128, 3, 1, 1)
			)

		self.decoder_3 = decoder_layer4(128, 64, 4)
		self.decoder_2 = decoder_layer4(64, 32, 4)
		self.decoder_1 = decoder_layer_out(32, 3, 5, get_feature=args.high_res)

	def forward(self, input_rgb, input_seg=None, neighbor_imgs=None):
		preds = []
		hidden_encon = []
		feature=None
		for scale in range(self.n_scales-1, -1, -1):
			scale = 1/(2**scale)
			input_ori = F.interpolate(input_rgb, scale_factor=scale, mode='bilinear', align_corners=True)
			input_pred = F.interpolate(preds[-1].detach(), scale_factor=2, mode='bilinear', align_corners=True) if scale!=1/(2**(self.n_scales-1)) else input_ori

			input = torch.cat([input_ori, input_pred], dim=1)

			encon1 = self.encoder_1(input)
			encon2 = self.encoder_2(encon1)
			encon3 = self.encoder_3(encon2)

			last_hidden_encon = F.interpolate(hidden_encon[-1], scale_factor=2, mode='bilinear', align_corners=True) if scale!=1/(2**(self.n_scales-1))\
										else encon3

			hidden_encon_input = torch.cat([encon3, last_hidden_encon], dim=1)
			encon3 = self.hidden_comb(hidden_encon_input)
			hidden_encon.append(encon3)

			decon2 = self.decoder_3(encon3)
			decon1 = self.decoder_2(encon2+decon2)
			if self.args.high_res:  
				pred, feature = self.decoder_1(encon1+decon1)
			else:
				pred = self.decoder_1(encon1+decon1)

			preds.append(pred)

		return preds, feature, None


class SRN4Seg(nn.Module):
	def __init__(self, args):
		super(SRN4Seg, self).__init__()
		self.n_scales = args.n_scales

		self.encoder_1 = nn.Sequential(
				nn.Conv2d(6+20, 32, 5, stride=1, padding=5//2),
				nn.LeakyReLU(0.2, inplace=True),
				ResnetBlock(32, 32, 5),
				ResnetBlock(32, 32, 5),
				ResnetBlock(32, 32, 5)          
			)
		self.encoder_2 = encoder_layer2(32, 64, 5)
		self.encoder_3 = encoder_layer2(64, 128, 5)

		self.hidden_comb = nn.Sequential(
				nn.Conv2d(256, 128, 3, 1, 1),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(128, 128, 3, 1, 1)
			)

		self.decoder_3 = decoder_layer4(128, 64, 4)
		self.decoder_2 = decoder_layer4(64, 32, 4)
		self.decoder_1 = decoder_layer_out(32, 3, 5)

	def forward(self, input_rgb, input_seg, neighbor_imgs=None):
		preds = []
		hidden_encon = []
		for scale in range(self.n_scales-1, -1, -1):
			scale = 1/(2**scale)
			input_ori = F.interpolate(input_rgb, scale_factor=scale, mode='bilinear', align_corners=True)
			input_seg_ori = F.interpolate(input_seg, scale_factor=scale, mode='bilinear', align_corners=True)
			input_pred = F.interpolate(preds[-1].detach(), scale_factor=2, mode='bilinear', align_corners=True) if scale!=1/(2**(self.n_scales-1)) else input_ori

			input = torch.cat([input_ori, input_pred, input_seg_ori], dim=1)

			encon1 = self.encoder_1(input)
			encon2 = self.encoder_2(encon1)
			encon3 = self.encoder_3(encon2)

			last_hidden_encon = F.interpolate(hidden_encon[-1], scale_factor=2, mode='bilinear', align_corners=True) if scale!=1/(2**(self.n_scales-1))\
										else encon3

			hidden_encon_input = torch.cat([encon3, last_hidden_encon], dim=1)
			encon3 = self.hidden_comb(hidden_encon_input)
			hidden_encon.append(encon3)

			decon2 = self.decoder_3(encon3)
			decon1 = self.decoder_2(encon2+decon2)
			pred = self.decoder_1(encon1+decon1)

			preds.append(pred)

		return preds


class SRN4Sharp(nn.Module):
	def __init__(self, args):
		super(SRN4Sharp, self).__init__()
		self.n_scales = args.n_scales
		self.args=args
		self.sharp_encoder_1 = nn.Sequential(
				nn.Conv2d(6, 32, 5, stride=1, padding=5//2),
				nn.LeakyReLU(0.2, inplace=True),
				ResnetBlock(32, 32, 5),
				ResnetBlock(32, 32, 5),
				ResnetBlock(32, 32, 5)          
			)
		self.sharp_encoder_2 = encoder_layer2(32, 64, 5)
		self.sharp_encoder_3 = encoder_layer2(64, 128, 5)


		self.encoder_1 = nn.Sequential(
				nn.Conv2d(6, 32, 5, stride=1, padding=5//2),
				nn.LeakyReLU(0.2, inplace=True),
				ResnetBlock(32, 32, 5),
				ResnetBlock(32, 32, 5),
				ResnetBlock(32, 32, 5)          
			)
		self.encoder_2 = encoder_layer2(32, 64, 5)
		self.encoder_3 = encoder_layer2(64, 128, 5)

		self.hidden_comb = nn.Sequential(
				nn.Conv2d(256, 128, 3, 1, 1),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(128, 128, 3, 1, 1)
			)

		self.decoder_3 = decoder_layer4(128, 64, 4)
		self.decoder_2 = decoder_layer4(64, 32, 4)
		self.decoder_1 = decoder_layer_out(32, 3, 5, get_feature=args.high_res)

	def forward(self, input_rgb, input_seg=None, neighbor_imgs=None):

		# encode neightbor imgs
		sharp_encon1 = self.sharp_encoder_1(neighbor_imgs)
		sharp_encon2 = self.sharp_encoder_2(sharp_encon1)
		sharp_encon3 = self.sharp_encoder_3(sharp_encon2)       
		feature=None
		preds = []
		hidden_encon = []
		for scale in range(self.n_scales-1, -1, -1):
			scale = 1/(2**scale)
			input_ori = F.interpolate(input_rgb, scale_factor=scale, mode='bilinear', align_corners=True)
			input_pred = F.interpolate(preds[-1].detach(), scale_factor=2, mode='bilinear', align_corners=True) if scale!=1/(2**(self.n_scales-1)) else input_ori

			input = torch.cat([input_ori, input_pred], dim=1)

			encon1 = self.encoder_1(input)
			encon2 = self.encoder_2(encon1)
			encon3 = self.encoder_3(encon2)

			last_hidden_encon = F.interpolate(hidden_encon[-1], scale_factor=2, mode='bilinear', align_corners=True) if scale!=1/(2**(self.n_scales-1))\
										else encon3

			hidden_encon_input = torch.cat([encon3, last_hidden_encon], dim=1)
			encon3 = self.hidden_comb(hidden_encon_input)
			hidden_encon.append(encon3)

			sharp_encon1_scaled = F.interpolate(sharp_encon1, scale_factor=scale, mode='bilinear', align_corners=True)
			sharp_encon2_scaled = F.interpolate(sharp_encon2, scale_factor=scale, mode='bilinear', align_corners=True)
			sharp_encon3_scaled = F.interpolate(sharp_encon3, scale_factor=scale, mode='bilinear', align_corners=True)

			decon2 = self.decoder_3(encon3 + sharp_encon3_scaled)
			decon1 = self.decoder_2(encon2+decon2 + sharp_encon2_scaled)
			if self.args.high_res:  
				pred, feature = self.decoder_1(encon1+decon1 + sharp_encon1_scaled)
			else:
				pred = self.decoder_1(encon1+decon1 + sharp_encon1_scaled)
			preds.append(pred)

		return preds, feature, None


class HResUnet(nn.Module):
	def __init__(self, args):
		super(HResUnet, self).__init__()

		self.head = nn.Sequential(
				nn.Conv2d(6, 32, 5, stride=1, padding=5//2),
				nn.LeakyReLU(0.2, inplace=True),
				ResnetBlock(32, 32, 5),
				ResnetBlock(32, 32, 5),
				ResnetBlock(32, 32, 5)          
			)
		self.encoder_1 = encoder_layer2(32, 32, 5)



		self.decoder_1 = decoder_layer4(32, 32, 4)
		self.tail = decoder_layer_out(32, 3, 5)

	def forward(self, refine_out, low_feature, input):
		x = self.head(input)
		encoded_1 = self.encoder_1(x)


		decoded_1 = self.decoder_1(encoded_1+low_feature)
		output = self.tail(decoded_1)

		return output


########### attention level 1 ############
class AttnRefine(nn.Module):
	def __init__(self, args):
		super(AttnRefine, self).__init__()
		# 128 * 256
		self.width = 64
		self.height = 32

		self.patch_width  = 1
		self.patch_height = 1

		self.stride_width  = 1
		self.stride_height = 1

		self.search_width  = 5
		self.search_height = 5

		self.cand_capacity = 1

		# some useful constant calculated from params above
		self.patch_h_num =  int(self.height/self.patch_height)
		self.patch_w_num =  int(self.width/self.patch_width)

		self.conv_encoder_1 = nn.Sequential(
				nn.Conv2d(3, 32, 3, stride=1, padding=1),  # 32, 128, 256
				nn.LeakyReLU(0.2, inplace=True)
			)
		self.conv_encoder_2 = nn.Sequential(
				nn.Conv2d(32, 64, 3, stride=2, padding=1), 
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(64, 64, 3, stride=1, padding=1), # 64, 64, 128
				nn.LeakyReLU(0.2, inplace=True)
			)
		self.conv_encoder_3 = nn.Sequential(
				nn.Conv2d(64, 64, 3, stride=2, padding=1),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(64, 64, 3, 1, padding=1)        # 128, 32, 64
			)


		self.conv_decoder_3 = nn.Sequential(
				nn.Conv2d(64*3, 128, 3, stride=1, padding=1),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(128, 64, 3, stride=1, padding=1),
				nn.LeakyReLU(0.2, inplace=True),
				ResnetBlock(64, 64, 3),
				ResnetBlock(64, 64, 3)
			)
		self.conv_decoder_2 = nn.Sequential(
				nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), 
				nn.Conv2d(64, 64, 3, stride=1, padding=1),              
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(64, 64, 3, stride=1, padding=1),              
				nn.LeakyReLU(0.2, inplace=True)                                 # 64, 64, 128
			)
		self.conv_decoder_1 = nn.Sequential(
				nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), 
				nn.Conv2d(64, 32, 3, stride=1, padding=1),                      
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(32, 32, 3, stride=1, padding=1),              
				nn.LeakyReLU(0.2, inplace=True)                                 # 32, 128, 256
			)

		self.conv_decoder_out = nn.Sequential(
				ResnetBlock(32, 32, 3),                     # 3, 128, 256
				nn.Conv2d(32, 3, 3, 1, 1)
			)


	def forward(self, coarse, seg=None, neighbors=None):
		cnt_start = time()
		bs = coarse.size(0)
		c=64
		pW = self.patch_width
		pH = self.patch_height
		sW = self.search_width
		sH = self.search_height
		nPH = self.patch_h_num
		nPW = self.patch_w_num 

		#### encoder ####
		coarse_enc_1 = self.conv_encoder_1(coarse)
		coarse_enc_2 = self.conv_encoder_2(coarse_enc_1)
		coarse_patches = self.conv_encoder_3(coarse_enc_2)
		neighbors_patches = torch.cat([
								self.conv_encoder_3(
									self.conv_encoder_2(
										self.conv_encoder_1(
											neighbors[:,:3]
											)
										)
									), 
								self.conv_encoder_3(
									self.conv_encoder_2(
										self.conv_encoder_1(
											neighbors[:,3:6]
											)
										)
									)], dim=1) 
		

		#### attention module ####
		# 1. extract window patches
		padded_neighbors_patchs = F.pad(neighbors_patches, pad=(sW//2, sW//2, sH//2, sH//2), value=1e-6)
		# (bs, 2*c*pH*pW, nPH+1, nPW+1)
		window_neighbors_patchs = padded_neighbors_patchs.unfold(2, sH, 1).unfold(3, sW, 1).permute(0,1,4,5,2,3).contiguous()
		window_neighbors_patchs = window_neighbors_patchs.view(bs, 2, c*pH*pW, sH, sW, nPH, nPW)
		resized_coarse_patches  = coarse_patches.view(bs, 1, c*pH*pW, 1, 1, nPH, nPW)

		# 2. compute corr map
		correlation_map = (window_neighbors_patchs*resized_coarse_patches).sum(dim=2)/window_neighbors_patchs.norm(dim=2)
		# (bs, 2, sH, sW, nPH, nPW)
		corr_softmax_map = F.softmax(correlation_map.view(bs, 2, sH*sW, nPH, nPW), dim=2)
		# (bs, 2, sH*sW, nPH, nPW)

		# 3. visualize attn map
		offset = torch.argmax(corr_softmax_map, dim=2)
		offset = torch.stack([offset//sW, offset % sH], dim=2).float() 
		# (bs, 2, 2, nPH, nPW)
		# h_add = torch.arange(nPH).view(1, 1, 1, nPH, 1).expand(bs, 1, 1, -1, nPW)#.cuda(correlation_map.get_device())
		# w_add = torch.arange(nPW).view(1, 1, 1, 1, nPW).expand(bs, 1, 1, nPH, -1)#.cuda(correlation_map.get_device())
		h_w_add = torch.zeros(bs, 1, 2, nPH, nPW).fill_(self.search_width//2)
		offset = offset.cpu() - h_w_add
		# flow =self.flow_to_image(offsets.numpy())


		corr_softmax_map = corr_softmax_map.unsqueeze(2)
		# (bs, 2, 1, sH*sW, nPH, nPW)

		# 3. attention, assign each patch in the window a probability and use them all
		# another idea is using argmax to track the patch
		neighbors_max_patch = (window_neighbors_patchs.view(bs, 2, c*pH*pW, sH*sW, nPH, nPW)*corr_softmax_map).sum(dim=3)\
									.view(bs, 2*c, pH, pW, nPH, nPW).permute(0,1,2,4,3,5).contiguous()\
										.view(bs, 2*c, self.height, self.width)             

		# decoder
		inputs = torch.cat([coarse_patches, neighbors_max_patch], dim=1)
		dec_3 = self.conv_decoder_3(inputs)
		dec_2 = self.conv_decoder_2(dec_3)
		dec_1 = self.conv_decoder_1(dec_2 + coarse_enc_2)
		out = self.conv_decoder_out(dec_1 + coarse_enc_1)

		# out = self.conv_tail(inputs) # bs, 3*16*16, 8, 16
		# out = out.view(bs, 3, 16, 16, 8, 16).permute(0,1,2,4,3,5).contiguous().view(bs, 3, 128, 256)#.contiguous()

		cnt_end = time()
		return [out], None, offset


class AttnBaseRefine(nn.Module):
	def __init__(self, args):
		super(AttnBaseRefine, self).__init__()
		# 128 * 256
		self.conv_encoder_1 = nn.Sequential(
				nn.Conv2d(3, 32, 3, stride=1, padding=1),  # 32, 128, 256           3, 1
				nn.LeakyReLU(0.2, inplace=True)
			)
		self.conv_encoder_2 = nn.Sequential(
				nn.Conv2d(32, 64, 3, stride=2, padding=1),                      #   5, 2
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(64, 64, 3, stride=1, padding=1), # 64, 64, 128        #   9, 2
				nn.LeakyReLU(0.2, inplace=True)
			)
		self.conv_encoder_3 = nn.Sequential(
				nn.Conv2d(64, 64, 3, stride=2, padding=1),                      #   13, 4
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(64, 64, 3, 1, padding=1)        # 128, 32, 64             21, 8
			)


		self.conv_decoder_3 = nn.Sequential(
				nn.Conv2d(64*3, 128, 3, stride=1, padding=1),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(128, 64, 3, stride=1, padding=1),
				nn.LeakyReLU(0.2, inplace=True),
				ResnetBlock(64, 64, 3),
				ResnetBlock(64, 64, 3)
			)
		self.conv_decoder_2 = nn.Sequential(
				nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), 
				nn.Conv2d(64, 64, 3, stride=1, padding=1),              
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(64, 64, 3, stride=1, padding=1),              
				nn.LeakyReLU(0.2, inplace=True)                                 # 64, 64, 128
			)
		self.conv_decoder_1 = nn.Sequential(
				nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), 
				nn.Conv2d(64, 32, 3, stride=1, padding=1),                      
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(32, 32, 3, stride=1, padding=1),              
				nn.LeakyReLU(0.2, inplace=True)                                 # 32, 128, 256
			)

		self.conv_decoder_out = nn.Sequential(
				ResnetBlock(32, 32, 3),                     # 3, 128, 256
				nn.Conv2d(32, 3, 3, 1, 1)
			)


	def forward(self, coarse, seg=None, neighbors=None):
		cnt_start = time()
		bs = coarse.size(0)
		c=64

		#### encoder ####
		coarse_enc_1 = self.conv_encoder_1(coarse)
		coarse_enc_2 = self.conv_encoder_2(coarse_enc_1)
		coarse_patches = self.conv_encoder_3(coarse_enc_2)
		neighbors_patches = torch.cat([
								self.conv_encoder_3(
									self.conv_encoder_2(
										self.conv_encoder_1(
											neighbors[:,:3]
											)
										)
									), 
								self.conv_encoder_3(
									self.conv_encoder_2(
										self.conv_encoder_1(
											neighbors[:,3:6]
											)
										)
									)], dim=1)              

		# decoder
		inputs = torch.cat([coarse_patches, neighbors_patches], dim=1)
		dec_3 = self.conv_decoder_3(inputs)
		dec_2 = self.conv_decoder_2(dec_3)
		dec_1 = self.conv_decoder_1(dec_2 + coarse_enc_2)
		out = self.conv_decoder_out(dec_1 + coarse_enc_1)

		# out = self.conv_tail(inputs) # bs, 3*16*16, 8, 16
		# out = out.view(bs, 3, 16, 16, 8, 16).permute(0,1,2,4,3,5).contiguous().view(bs, 3, 128, 256)#.contiguous()

		cnt_end = time()
		return [out], None, None


########### attention level 2 ###############
class AttnRefineV2(nn.Module):
	def __init__(self, args):
		super(AttnRefineV2, self).__init__()
		resnet101 = torchvision.models.resnet101(pretrained=True)
		self.resnet101 = my_resnet101(resnet101)
		for param in self.resnet101.parameters():
			param.requires_grad = False

		self.encoder_1 = nn.Sequential(
				nn.Conv2d(3, 32, 3, stride=1, padding=3//2),
				nn.LeakyReLU(0.2, inplace=True),
				ResnetBlock(32, 32, 3),
				ResnetBlock(32, 32, 3),
				ResnetBlock(32, 32, 3)          
			)
		self.encoder_2 = encoder_layer3(32, 64, 3)
		self.encoder_3 = encoder_layer3(64, 128, 3) # 32*64
		self.encoder_4 = encoder_layer3(128, 128, 3) # 16*32

		self.mid = nn.Sequential(
				nn.Conv2d(128*3, 128, 3, stride=1, padding=1),
				nn.LeakyReLU(0.2, inplace=True),
				ResnetBlock(128, 128, 3),
				ResnetBlock(128, 128, 3)			
			)

		self.decoder_4 = decoder_layer5(128, 128, 4)
		self.decoder_3 = decoder_layer5(128, 64, 4)
		self.decoder_2 = decoder_layer5(64, 32, 4)
		self.decoder_1 = decoder_layer_out(32, 3, 3)

		self.w = 9
		self.h = 5


	def corrmap(self, x, t1, t2):
		bs, c, H, W = x.size() # H = 16, W = 32
		# x_pad = F.pad(x, pad=(self.w//2, self.w//2, self.h//2, self.h//2), value=0)
		t = torch.cat([t1.unsqueeze(1), t2.unsqueeze(1)], dim=1) # bs, 2, c, H, W
		t_pad = F.pad(t, pad=(self.w//2, self.w//2, self.h//2, self.h//2), value=-100) # set to -100 to enlarge distance

		t_nns = t_pad.unfold(dimension=3, size=self.h, step=1)# bs, 2, c, H, W, h
		t_nns = t_nns.unfold(dimension=4, size=self.w, step=1)# bs, 2, c, H, W, h, w
		assert t_nns.size()[3:5] == (H, W), t_nns.size()

		x_tf = x.view(bs, 1, c, H, W, 1, 1)

		dis_map = torch.sum((t_nns - x_tf)**2, dim=2) # bs, 2, H, W, h, w
		dis_map_1d = dis_map.view(bs, 2, H, W, self.h*self.w) # bs, 2, H, W, h*w

		# calculate flow_map
		flow_map = dis_map_1d.argmin(dim=4) 
		# print("max", (flow_map//self.h).max())
		# print("min", (flow_map%self.h).min())
		flow_map = torch.stack([flow_map//self.h, flow_map%self.h], dim=2).float() # bs, 2, 2, H, W      w, h




		h_w_add = torch.zeros(bs, 1, 2, H, W)
		h_w_add[:,:,0,:,:] = self.w//2
		h_w_add[:,:,1,:,:] = self.h//2
		flow_map = flow_map.cpu() - h_w_add

		# use soft attention
		sim_map = 1./(dis_map_1d+1e-6) # bs, 2, H, W, h*w
		prob_map = F.softmax(sim_map, dim=4) # bs, 2, H, W, h*w

		return prob_map, flow_map

	def forward(self, x, neighbors):

		img1 = neighbors[:,:3]
		img2 = neighbors[:,3:6] 
		x_f1, x_f2, x_f3 = self.resnet101(x)  # 16*32
		x_cat_f = torch.cat([
				x_f1
				# F.interpolate(x_f2, scale_factor=2, mode='bilinear', align_corners=True) 
				# F.interpolate(x_f3, scale_factor=4, mode='bilinear', align_corners=True) # this part
			], dim=1)
		img1_f1, img1_f2, img1_f3 = self.resnet101(img1)
		img1_cat_f = torch.cat([
				img1_f1
				# F.interpolate(img1_f2, scale_factor=2, mode='bilinear', align_corners=True)
				# F.interpolate(img1_f3, scale_factor=4, mode='bilinear', align_corners=True)
			], dim=1)
		img2_f1, img2_f2, img2_f3 = self.resnet101(img2)
		img2_cat_f = torch.cat([
				img2_f1
				# F.interpolate(img2_f2, scale_factor=2, mode='bilinear', align_corners=True)
				# F.interpolate(img2_f3, scale_factor=4, mode='bilinear', align_corners=True)
			], dim=1)

		prob_map, flow_map = self.corrmap(x_cat_f, img1_cat_f, img2_cat_f)

		x_enc1 = self.encoder_1(x)
		x_enc2 = self.encoder_2(x_enc1)
		x_enc3 = self.encoder_3(x_enc2)
		x_enc4 = self.encoder_4(x_enc3)

		img1_enc1 = self.encoder_1(img1)
		img1_enc2 = self.encoder_2(img1_enc1)
		img1_enc3 = self.encoder_3(img1_enc2)
		img1_enc4 = self.encoder_4(img1_enc3)

		img2_enc1 = self.encoder_1(img2)
		img2_enc2 = self.encoder_2(img2_enc1)
		img2_enc3 = self.encoder_3(img2_enc2)
		img2_enc4 = self.encoder_4(img2_enc3)

		# enc4 feature fusion
		enc4_fusion = torch.cat([img1_enc4.unsqueeze(1), img2_enc4.unsqueeze(1)], dim=1)
		enc4_fusion_pad = F.pad(enc4_fusion, pad=(self.w//2, self.w//2, self.h//2, self.h//2), value=0)
		enc4_fusion_patch = enc4_fusion_pad.unfold(dimension=3, size=self.h, step=1).unfold(dimension=4, size=self.w, step=1)# bs, 2, c, H, W, h, w
		re_shape = list(enc4_fusion_patch.size()[:5]) + [self.h*self.w]
		enc4_fusion_patch_weighted = (enc4_fusion_patch.contiguous().view(re_shape).contiguous()*prob_map.unsqueeze(2)).\
											sum(dim=5) # bs, 2, c, H, W

		mid_in = torch.cat([x_enc4, enc4_fusion_patch_weighted[:,0], enc4_fusion_patch_weighted[:, 1]], dim=1)
		dec4_in = self.mid(mid_in) 
		dec4_out = self.decoder_4(dec4_in)
		dec3_out = self.decoder_3(dec4_out+x_enc3)
		dec2_out = self.decoder_2(dec3_out+x_enc2)
		dec1_out = self.decoder_1(dec2_out+x_enc1)

		return dec1_out, flow_map


class AttnRefineV2O(nn.Module):
	def __init__(self, args):
		super(AttnRefineV2O, self).__init__()
		self.encoder_1 = nn.Sequential(
				nn.Conv2d(3, 32, 3, stride=1, padding=3//2),
				nn.LeakyReLU(0.2, inplace=True),
				ResnetBlock(32, 32, 3),
				ResnetBlock(32, 32, 3),
				ResnetBlock(32, 32, 3)          
			)
		self.encoder_2 = encoder_layer3(32, 64, 3)
		self.encoder_3 = encoder_layer3(64, 128, 3) # 32*64
		self.encoder_4 = encoder_layer3(128, 128, 3) # 16*32

		self.mid = nn.Sequential(
				nn.Conv2d(128*3, 128, 3, stride=1, padding=1),
				nn.LeakyReLU(0.2, inplace=True),
				ResnetBlock(128, 128, 3),
				ResnetBlock(128, 128, 3)			
			)

		self.decoder_4 = decoder_layer5(128, 128, 4)
		self.decoder_3 = decoder_layer5(128, 64, 4)
		self.decoder_2 = decoder_layer5(64, 32, 4)
		self.decoder_1 = decoder_layer_out(32, 3, 3)

		self.w = 9
		self.h = 5


	def corrmap(self, x, t1, t2):
		bs, c, H, W = x.size() # H = 16, W = 32
		# x_pad = F.pad(x, pad=(self.w//2, self.w//2, self.h//2, self.h//2), value=0)
		t = torch.cat([t1.unsqueeze(1), t2.unsqueeze(1)], dim=1) # bs, 2, c, H, W
		t_pad = F.pad(t, pad=(self.w//2, self.w//2, self.h//2, self.h//2), value=-100) # set to -100 to enlarge distance

		t_nns = t_pad.unfold(dimension=3, size=self.h, step=1)# bs, 2, c, H, W, h
		t_nns = t_nns.unfold(dimension=4, size=self.w, step=1)# bs, 2, c, H, W, h, w
		assert t_nns.size()[3:5] == (H, W), t_nns.size()

		x_tf = x.view(bs, 1, c, H, W, 1, 1)

		dis_map = torch.sum((t_nns - x_tf)**2, dim=2) # bs, 2, H, W, h, w
		dis_map_1d = dis_map.view(bs, 2, H, W, self.h*self.w) # bs, 2, H, W, h*w

		# calculate flow_map
		flow_map = dis_map_1d.argmin(dim=4) 
		# print("max", (flow_map//self.h).max())
		# print("min", (flow_map%self.h).min())
		flow_map = torch.stack([flow_map//self.h, flow_map%self.h], dim=2).float() # bs, 2, 2, H, W      w, h




		h_w_add = torch.zeros(bs, 1, 2, H, W)
		h_w_add[:,:,0,:,:] = self.w//2
		h_w_add[:,:,1,:,:] = self.h//2
		flow_map = flow_map.cpu() - h_w_add

		# use soft attention
		sim_map = 1./(dis_map_1d+1e-6) # bs, 2, H, W, h*w
		prob_map = F.softmax(sim_map, dim=4) # bs, 2, H, W, h*w

		return prob_map, flow_map

	def forward(self, x, neighbors):

		img1 = neighbors[:,:3]
		img2 = neighbors[:,3:6] 
		x_enc1 = self.encoder_1(x)
		x_enc2 = self.encoder_2(x_enc1)
		x_enc3 = self.encoder_3(x_enc2)
		x_enc4 = self.encoder_4(x_enc3)

		img1_enc1 = self.encoder_1(img1)
		img1_enc2 = self.encoder_2(img1_enc1)
		img1_enc3 = self.encoder_3(img1_enc2)
		img1_enc4 = self.encoder_4(img1_enc3)

		img2_enc1 = self.encoder_1(img2)
		img2_enc2 = self.encoder_2(img2_enc1)
		img2_enc3 = self.encoder_3(img2_enc2)
		img2_enc4 = self.encoder_4(img2_enc3)


		prob_map, flow_map = self.corrmap(x_enc4, img1_enc4, img2_enc4)

		# enc4 feature fusion
		enc4_fusion = torch.cat([img1_enc4.unsqueeze(1), img2_enc4.unsqueeze(1)], dim=1)
		enc4_fusion_pad = F.pad(enc4_fusion, pad=(self.w//2, self.w//2, self.h//2, self.h//2), value=0)
		enc4_fusion_patch = enc4_fusion_pad.unfold(dimension=3, size=self.h, step=1).unfold(dimension=4, size=self.w, step=1)# bs, 2, c, H, W, h, w
		re_shape = list(enc4_fusion_patch.size()[:5]) + [self.h*self.w]
		enc4_fusion_patch_weighted = (enc4_fusion_patch.contiguous().view(re_shape).contiguous()*prob_map.unsqueeze(2)).\
											sum(dim=5) # bs, 2, c, H, W

		mid_in = torch.cat([x_enc4, enc4_fusion_patch_weighted[:,0], enc4_fusion_patch_weighted[:, 1]], dim=1)
		dec4_in = self.mid(mid_in) 
		dec4_out = self.decoder_4(dec4_in)
		dec3_out = self.decoder_3(dec4_out+x_enc3)
		dec2_out = self.decoder_2(dec3_out+x_enc2)
		dec1_out = self.decoder_1(dec2_out+x_enc1)

		return dec1_out, flow_map


class AttnRefineV2Base(nn.Module):
	def __init__(self, args):
		super(AttnRefineV2Base, self).__init__()
		# resnet101 = torchvision.models.resnet101(pretrained=True)
		# self.resnet101 = my_resnet101(resnet101)
		# for param in self.resnet101.parameters():
			# param.requires_grad = False

		self.encoder_1 = nn.Sequential(
				nn.Conv2d(3, 32, 3, stride=1, padding=3//2),
				nn.LeakyReLU(0.2, inplace=True),
				ResnetBlock(32, 32, 3),
				ResnetBlock(32, 32, 3),
				ResnetBlock(32, 32, 3)          
			)
		self.encoder_2 = encoder_layer3(32, 64, 3)
		self.encoder_3 = encoder_layer3(64, 128, 3) # 32*64
		self.encoder_4 = encoder_layer3(128, 128, 3) # 16*32

		self.mid = nn.Sequential(
				nn.Conv2d(128*3, 128, 3, stride=1, padding=1),
				nn.LeakyReLU(0.2, inplace=True),
				ResnetBlock(128, 128, 3),
				ResnetBlock(128, 128, 3)			
			)

		self.decoder_4 = decoder_layer5(128, 128, 4)
		self.decoder_3 = decoder_layer5(128, 64, 4)
		self.decoder_2 = decoder_layer5(64, 32, 4)
		self.decoder_1 = decoder_layer_out(32, 3, 3)

		self.w = 9
		self.h = 5
		self.W = 32
		self.H = 16


	def forward(self, x, neighbors):

		img1 = neighbors[:,:3]
		img2 = neighbors[:,3:6] 
		

		x_enc1 = self.encoder_1(x)
		x_enc2 = self.encoder_2(x_enc1)
		x_enc3 = self.encoder_3(x_enc2)
		x_enc4 = self.encoder_4(x_enc3)

		img1_enc1 = self.encoder_1(img1)
		img1_enc2 = self.encoder_2(img1_enc1)
		img1_enc3 = self.encoder_3(img1_enc2)
		img1_enc4 = self.encoder_4(img1_enc3)

		img2_enc1 = self.encoder_1(img2)
		img2_enc2 = self.encoder_2(img2_enc1)
		img2_enc3 = self.encoder_3(img2_enc2)
		img2_enc4 = self.encoder_4(img2_enc3)


		mid_in = torch.cat([x_enc4, img1_enc4, img2_enc4], dim=1)
		dec4_in = self.mid(mid_in) 
		dec4_out = self.decoder_4(dec4_in)
		dec3_out = self.decoder_3(dec4_out+x_enc3)
		dec2_out = self.decoder_2(dec3_out+x_enc2)
		dec1_out = self.decoder_1(dec2_out+x_enc1)

		return dec1_out, None


########### attention level 3 ###############
class AttnRefineV3(nn.Module):
	def __init__(self, args):
		super(AttnRefineV3, self).__init__()
		self.attn_encoder_1 = nn.Sequential(
				nn.Conv2d(3, 32, 3, stride=1, padding=3//2),
				nn.LeakyReLU(0.2, inplace=True),
				ResnetBlock(32, 32, 3),
				ResnetBlock(32, 32, 3),
				ResnetBlock(32, 32, 3)          
			)
		self.attn_encoder_2 = encoder_layer3(32, 64, 3)	# 64 * 128
		self.attn_encoder_3 = encoder_layer3(64, 64, 3)	# 32 * 64

		self.img_encoder_1 = nn.Sequential(
				nn.Conv2d(3, 32, 3, stride=1, padding=3//2),
				nn.LeakyReLU(0.2, inplace=True),
				ResnetBlock(32, 32, 3),
				ResnetBlock(32, 32, 3),
				ResnetBlock(32, 32, 3)          
			)
		self.img_encoder_2 = encoder_layer3(32, 64, 3)	# 64 * 128
		self.img_encoder_3 = encoder_layer3(64, 64, 3)	# 32 * 64

		self.mid = nn.Sequential(
				nn.Conv2d(64*3, 128, 3, stride=1, padding=1),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(128, 64, 3, stride=1, padding=1),
				nn.LeakyReLU(0.2, inplace=True),
				ResnetBlock(64, 64, 3),
				ResnetBlock(64, 64, 3)			
			)

		self.decoder_3 = decoder_layer5(64, 64, 4)
		self.decoder_2 = decoder_layer5(64, 32, 4)
		self.decoder_1 = decoder_layer_out(32, 3, 3)

		self.w = 17
		self.h = 9
		self.W = 64
		self.H = 32


	def corrmap(self, x, t1, t2):
		bs, c, H, W = x.size() # H = 32, W = 64

		x_normed  = x/x.norm(dim=1, keepdim=True)
		t1_normed = t1/t1.norm(dim=1, keepdim=True)
		t2_normed = t2/t2.norm(dim=1, keepdim=True)

		t = torch.cat([t1_normed.unsqueeze(1), t2_normed.unsqueeze(1)], dim=1) # bs, 2, c, H, W
		t_pad = F.pad(t, pad=(self.w//2, self.w//2, self.h//2, self.h//2), value=-100) # set to -100 to enlarge distance

		t_nns = t_pad.unfold(dimension=3, size=self.h, step=1)# bs, 2, c, H, W, h
		t_nns = t_nns.unfold(dimension=4, size=self.w, step=1)# bs, 2, c, H, W, h, w
		assert t_nns.size()[3:5] == (H, W), t_nns.size()

		x_tf = x_normed.view(bs, 1, c, H, W, 1, 1)

		sim_map = torch.sum(t_nns*x_tf, dim=2) # bs, 2, H, W, h, w
		sim_map_1d = sim_map.view(bs, 2, H, W, self.h*self.w) # bs, 2, H, W, h*w

		# calculate flow_map
		flow_map = sim_map_1d.argmax(dim=4) 
		flow_map = torch.stack([flow_map//self.h, flow_map%self.h], dim=2).float() # bs, 2, 2, H, W      w, h

		h_w_add = torch.zeros(bs, 1, 2, H, W)
		h_w_add[:,:,0,:,:] = self.w//2
		h_w_add[:,:,1,:,:] = self.h//2
		flow_map = flow_map.cpu() - h_w_add

		prob_map = F.softmax(sim_map_1d, dim=4) # bs, 2, H, W, h*w

		return prob_map, flow_map

	def forward(self, x, neighbors):

		img1 = neighbors[:,:3]
		img2 = neighbors[:,3:6] 

		x_attn_enc1 = self.attn_encoder_1(x)
		x_attn_enc2 = self.attn_encoder_2(x_attn_enc1)
		x_attn_enc3 = self.attn_encoder_3(x_attn_enc2)

		img1_attn_enc1 = self.attn_encoder_1(img1)
		img1_attn_enc2 = self.attn_encoder_2(img1_attn_enc1)
		img1_attn_enc3 = self.attn_encoder_3(img1_attn_enc2)

		img2_attn_enc1 = self.attn_encoder_1(img2)
		img2_attn_enc2 = self.attn_encoder_2(img2_attn_enc1)
		img2_attn_enc3 = self.attn_encoder_3(img2_attn_enc2)

		prob_map, flow_map = self.corrmap(x_attn_enc3, img1_attn_enc3, img2_attn_enc3)


		x_enc1 = self.img_encoder_1(x)
		x_enc2 = self.img_encoder_2(x_enc1)
		x_enc3 = self.img_encoder_3(x_enc2)

		img1_enc1 = self.img_encoder_1(img1)
		img1_enc2 = self.img_encoder_2(img1_enc1)
		img1_enc3 = self.img_encoder_3(img1_enc2)

		img2_enc1 = self.img_encoder_1(img2)
		img2_enc2 = self.img_encoder_2(img2_enc1)
		img2_enc3 = self.img_encoder_3(img2_enc2)

		# enc4 feature fusion
		enc3_fusion = torch.cat([img1_enc3.unsqueeze(1), img2_enc3.unsqueeze(1)], dim=1)
		enc3_fusion_pad = F.pad(enc3_fusion, pad=(self.w//2, self.w//2, self.h//2, self.h//2), value=0)
		enc3_fusion_patch = enc3_fusion_pad.unfold(dimension=3, size=self.h, step=1).unfold(dimension=4, size=self.w, step=1)# bs, 2, c, H, W, h, w
		re_shape = list(enc3_fusion_patch.size()[:5]) + [self.h*self.w]
		enc3_fusion_patch_weighted = (enc3_fusion_patch.contiguous().view(re_shape).contiguous()*prob_map.unsqueeze(2)).\
											sum(dim=5) # bs, 2, c, H, W

		mid_in = torch.cat([x_enc3, enc3_fusion_patch_weighted[:,0], enc3_fusion_patch_weighted[:, 1]], dim=1)
		dec3_in = self.mid(mid_in) 
		dec3_out = self.decoder_3(dec3_in)
		dec2_out = self.decoder_2(dec3_out)
		dec1_out = self.decoder_1(dec2_out)

		return dec1_out, flow_map


########### attention level 3 ###############
class AttnRefineV3Base(nn.Module):
	def __init__(self, args):
		super(AttnRefineV3Base, self).__init__()
		self.img_encoder_1 = nn.Sequential(
				nn.Conv2d(3, 32, 3, stride=1, padding=3//2),
				nn.LeakyReLU(0.2, inplace=True),
				ResnetBlock(32, 32, 3),
				ResnetBlock(32, 32, 3),
				ResnetBlock(32, 32, 3)          
			)
		self.img_encoder_2 = encoder_layer3(32, 64, 3)	# 64 * 128
		self.img_encoder_3 = encoder_layer3(64, 128, 3)	# 32 * 64

		self.mid = nn.Sequential(
				nn.Conv2d(128*3, 128, 3, stride=1, padding=1),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(128, 128, 3, stride=1, padding=1),
				nn.LeakyReLU(0.2, inplace=True),
				ResnetBlock(128, 128, 3),
				ResnetBlock(128, 128, 3)			
			)

		self.decoder_3 = decoder_layer5(128, 64, 4)
		self.decoder_2 = decoder_layer5(64, 32, 4)
		self.decoder_1 = decoder_layer_out(32, 3, 3)

		self.w = 17
		self.h = 9
		self.W = 64
		self.H = 32

	def forward(self, x, neighbors):

		img1 = neighbors[:,:3]
		img2 = neighbors[:,3:6] 

		x_enc1 = self.img_encoder_1(x)
		x_enc2 = self.img_encoder_2(x_enc1)
		x_enc3 = self.img_encoder_3(x_enc2)

		img1_enc1 = self.img_encoder_1(img1)
		img1_enc2 = self.img_encoder_2(img1_enc1)
		img1_enc3 = self.img_encoder_3(img1_enc2)

		img2_enc1 = self.img_encoder_1(img2)
		img2_enc2 = self.img_encoder_2(img2_enc1)
		img2_enc3 = self.img_encoder_3(img2_enc2)

		mid_in = torch.cat([x_enc3, img1_enc3, img2_enc3], dim=1)
		dec3_in = self.mid(mid_in) 
		dec3_out = self.decoder_3(dec3_in)
		dec2_out = self.decoder_2(dec3_out)
		dec1_out = self.decoder_1(dec2_out)

		return dec1_out, None


########### attention level 3 ###############
class AttnRefineV4(nn.Module):
	def __init__(self, args):
		super(AttnRefineV4, self).__init__()
		self.attn_encoder_1 = nn.Sequential(
				nn.Conv2d(3, 64, 3, stride=1, padding=3//2),
				nn.LeakyReLU(0.2, inplace=True),
				ResnetBlock(64, 64, 3)
			)
		self.attn_encoder_2 = encoder_layer3(64, 128, 3)	# 64 * 128

		self.img_encoder_1 = nn.Sequential(
				nn.Conv2d(3, 32, 3, stride=1, padding=3//2),
				nn.LeakyReLU(0.2, inplace=True),
				ResnetBlock(32, 32, 3),
				ResnetBlock(32, 32, 3),
				ResnetBlock(32, 32, 3)          
			)
		self.img_encoder_2 = encoder_layer3(32, 64, 3)	# 64 * 128
		self.img_encoder_3 = encoder_layer3(64, 64, 3)	# 32 * 64

		self.mid = nn.Sequential(
				nn.Conv2d(64*3, 128, 3, stride=1, padding=1),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(128, 64, 3, stride=1, padding=1),
				nn.LeakyReLU(0.2, inplace=True),
				ResnetBlock(64, 64, 3),
				ResnetBlock(64, 64, 3)			
			)

		self.decoder_3 = decoder_layer5(64, 64, 4)
		self.decoder_2 = decoder_layer5(64, 32, 4)
		self.decoder_1 = decoder_layer_out(32, 3, 3)

		self.w = 17
		self.h = 9
		self.W = 64
		self.H = 32


	def corrmap(self, x, t1, t2):
		bs, c, H, W = x.size() # H = 32, W = 64

		x_normed  = x/x.norm(dim=1, keepdim=True)
		t1_normed = t1/t1.norm(dim=1, keepdim=True)
		t2_normed = t2/t2.norm(dim=1, keepdim=True)

		t = torch.cat([t1_normed.unsqueeze(1), t2_normed.unsqueeze(1)], dim=1) # bs, 2, c, H, W
		t_pad = F.pad(t, pad=(self.w//2, self.w//2, self.h//2, self.h//2), value=-100) # set to -100 to enlarge distance

		t_nns = t_pad.unfold(dimension=3, size=self.h, step=1)# bs, 2, c, H, W, h
		t_nns = t_nns.unfold(dimension=4, size=self.w, step=1)# bs, 2, c, H, W, h, w
		assert t_nns.size()[3:5] == (H, W), t_nns.size()

		x_tf = x_normed.view(bs, 1, c, H, W, 1, 1)

		sim_map = torch.sum(t_nns*x_tf, dim=2) # bs, 2, H, W, h, w
		sim_map_1d = sim_map.view(bs, 2, H, W, self.h*self.w) # bs, 2, H, W, h*w

		# calculate flow_map
		flow_map = sim_map_1d.argmax(dim=4) 
		flow_map = torch.stack([flow_map//self.h, flow_map%self.h], dim=2).float() # bs, 2, 2, H, W      w, h

		h_w_add = torch.zeros(bs, 1, 2, H, W)
		h_w_add[:,:,0,:,:] = self.w//2
		h_w_add[:,:,1,:,:] = self.h//2
		flow_map = flow_map.cpu() - h_w_add

		prob_map = F.softmax(sim_map_1d, dim=4) # bs, 2, H, W, h*w

		return prob_map, flow_map

	def forward(self, x, neighbors):

		img1 = neighbors[:,:3]
		img2 = neighbors[:,3:6] 

		x_attn_enc1 = self.attn_encoder_1(x)
		x_attn_enc2 = self.attn_encoder_2(x_attn_enc1)
		x_attn_enc3 = self.attn_encoder_3(x_attn_enc2)

		img1_attn_enc1 = self.attn_encoder_1(img1)
		img1_attn_enc2 = self.attn_encoder_2(img1_attn_enc1)
		img1_attn_enc3 = self.attn_encoder_3(img1_attn_enc2)

		img2_attn_enc1 = self.attn_encoder_1(img2)
		img2_attn_enc2 = self.attn_encoder_2(img2_attn_enc1)
		img2_attn_enc3 = self.attn_encoder_3(img2_attn_enc2)

		prob_map, flow_map = self.corrmap(x_attn_enc3, img1_attn_enc3, img2_attn_enc3)


		x_enc1 = self.img_encoder_1(x)
		x_enc2 = self.img_encoder_2(x_enc1)
		x_enc3 = self.img_encoder_3(x_enc2)

		img1_enc1 = self.img_encoder_1(img1)
		img1_enc2 = self.img_encoder_2(img1_enc1)
		img1_enc3 = self.img_encoder_3(img1_enc2)

		img2_enc1 = self.img_encoder_1(img2)
		img2_enc2 = self.img_encoder_2(img2_enc1)
		img2_enc3 = self.img_encoder_3(img2_enc2)

		# enc4 feature fusion
		enc3_fusion = torch.cat([img1_enc3.unsqueeze(1), img2_enc3.unsqueeze(1)], dim=1)
		enc3_fusion_pad = F.pad(enc3_fusion, pad=(self.w//2, self.w//2, self.h//2, self.h//2), value=0)
		enc3_fusion_patch = enc3_fusion_pad.unfold(dimension=3, size=self.h, step=1).unfold(dimension=4, size=self.w, step=1)# bs, 2, c, H, W, h, w
		re_shape = list(enc3_fusion_patch.size()[:5]) + [self.h*self.w]
		enc3_fusion_patch_weighted = (enc3_fusion_patch.contiguous().view(re_shape).contiguous()*prob_map.unsqueeze(2)).\
											sum(dim=5) # bs, 2, c, H, W

		mid_in = torch.cat([x_enc3, enc3_fusion_patch_weighted[:,0], enc3_fusion_patch_weighted[:, 1]], dim=1)
		dec3_in = self.mid(mid_in) 
		dec3_out = self.decoder_3(dec3_in)
		dec2_out = self.decoder_2(dec3_out)
		dec1_out = self.decoder_1(dec2_out)

		return dec1_out, flow_map

########### attention level 3 ###############
class AttnRefineV4Base(nn.Module):
	def __init__(self, args):
		super(AttnRefineV4Base, self).__init__()
		self.img_encoder_1 = nn.Sequential(
				nn.Conv2d(3, 32, 3, stride=1, padding=3//2),
				nn.LeakyReLU(0.2, inplace=True),
				ResnetBlock(32, 32, 3),
				ResnetBlock(32, 32, 3),
				ResnetBlock(32, 32, 3)          
			)
		self.img_encoder_2 = encoder_layer3(32, 64, 3)	# 64 * 128
		self.img_encoder_3 = encoder_layer3(64, 128, 3)	# 32 * 64

		self.mid = nn.Sequential(
				nn.Conv2d(128*3, 128, 3, stride=1, padding=1),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(128, 128, 3, stride=1, padding=1),
				nn.LeakyReLU(0.2, inplace=True),
				ResnetBlock(128, 128, 3),
				ResnetBlock(128, 128, 3)			
			)

		self.decoder_3 = decoder_layer5(128, 64, 4)
		self.decoder_2 = decoder_layer5(64, 32, 4)
		self.decoder_1 = decoder_layer_out(32, 3, 3)

		self.w = 17
		self.h = 9
		self.W = 64
		self.H = 32

	def forward(self, x, neighbors):

		img1 = neighbors[:,:3]
		img2 = neighbors[:,3:6] 

		x_enc1 = self.img_encoder_1(x)
		x_enc2 = self.img_encoder_2(x_enc1)
		x_enc3 = self.img_encoder_3(x_enc2)

		img1_enc1 = self.img_encoder_1(img1)
		img1_enc2 = self.img_encoder_2(img1_enc1)
		img1_enc3 = self.img_encoder_3(img1_enc2)

		img2_enc1 = self.img_encoder_1(img2)
		img2_enc2 = self.img_encoder_2(img2_enc1)
		img2_enc3 = self.img_encoder_3(img2_enc2)

		mid_in = torch.cat([x_enc3, img1_enc3, img2_enc3], dim=1)
		dec3_in = self.mid(mid_in) 
		dec3_out = self.decoder_3(dec3_in)
		dec2_out = self.decoder_2(dec3_out)
		dec1_out = self.decoder_1(dec2_out)

		return dec1_out, None


########### multiscale conv ############
class MSConv2d(nn.Module):
	def __init__(self, in_dim, out_dim, n_sc, kss, stride, act=False):
		super(MSConv2d, self).__init__()
		self.in_dim     = in_dim
		self.out_dim    = out_dim
		self.n_sc       = n_sc
		self.kss        = [kss]*n_sc if type(kss) == int else kss 
		self.stride     = stride

		assert self.n_sc == len(self.kss), '# scales must = # kernel sizes'

		for i in range(self.n_sc):
			setattr(self, 'conv'+str(i),
					nn.Sequential(
						nn.AvgPool2d(2**i),
						nn.Conv2d(self.in_dim, self.out_dim, self.kss[i], stride=self.stride, padding=self.kss[i]//2),
						nn.LeakyReLU(0.2, inplace=True),
						nn.Upsample(scale_factor=2**i, mode='bilinear', align_corners=True)
					)if act else \
					nn.Sequential(
						nn.AvgPool2d(2**i),
						nn.Conv2d(self.in_dim, self.out_dim, self.kss[i], stride=self.stride, padding=self.kss[i]//2),
						nn.Upsample(scale_factor=2**i, mode='bilinear', align_corners=True)
					)
				)

	def forward(self, input):
		bs, c, h, w = input.size()
		conv_outs = 0
		for i in range(self.n_sc):
			conv_out = getattr(self, "conv"+str(i))(input)
			conv_outs += conv_out
		return conv_outs


class MSResnetBlock(nn.Module):
	def __init__(self, dim, n_sc, kss):
		super(MSResnetBlock, self).__init__()
		self.conv = nn.Sequential(
				MSConv2d(dim,  dim, n_sc, kss, 1, True),
				MSConv2d(dim,  dim, n_sc, kss, 1, False)
			)

	def forward(self, input):
		conv_out = self.conv(input)
		return  conv_out + input


class MSBaseRefine(nn.Module):
	def __init__(self, args):
		super(MSBaseRefine, self).__init__()
		# 128 * 256
		# encoder
		self.conv_encoder_1 = nn.Sequential(
				MSConv2d(3, 32, 3, 3, 1, True),  # 32, 128, 256         3, 1
			)
		self.conv_encoder_2 = nn.Sequential(
				MSConv2d(32, 64, 3, 3, 2, True),                        #   5, 2
				MSConv2d(64, 64, 3, 3, 1, True)     # 64, 64, 128       #   9, 2
			)
		self.conv_encoder_3 = nn.Sequential(
				MSConv2d(64, 64, 3, 3, 2, True),                        #   13, 4
				MSConv2d(64, 64, 3, 3, 1, False)        # 128, 32, 64               21, 8
			)

		# bottleneck transform
		self.conv_decoder_3 = nn.Sequential(
				MSConv2d(64*3, 128, 3, 3, 1, True),
				MSConv2d(128, 64, 3, 3, 1, True),
				MSResnetBlock(64, 3, 3),
				MSResnetBlock(64, 3, 3)
			)

		# decoder
		self.conv_decoder_2 = nn.Sequential(
				nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), 
				MSConv2d(64, 64, 3, 3, 1, True),              
				MSConv2d(64, 64, 3, 3, 1, True)                                 # 64, 64, 128
			)
		self.conv_decoder_1 = nn.Sequential(
				nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), 
				MSConv2d(64, 32, 3, 3, 1, True),
				MSConv2d(32, 32, 3, 3, 1, True)                                 # 32, 128, 256
			)

		self.conv_decoder_out = nn.Sequential(
				MSResnetBlock(32, 3, 3),                     # 3, 128, 256
				nn.Conv2d(32, 3, 3, 1, 1)
			)


	def forward(self, coarse, seg=None, neighbors=None):
		cnt_start = time()
		bs = coarse.size(0)
		c=64

		#### encoder ####
		coarse_enc_1 = self.conv_encoder_1(coarse)
		coarse_enc_2 = self.conv_encoder_2(coarse_enc_1)
		coarse_patches = self.conv_encoder_3(coarse_enc_2)
		neighbors_patches = torch.cat([
								self.conv_encoder_3(
									self.conv_encoder_2(
										self.conv_encoder_1(
											neighbors[:,:3]
											)
										)
									), 
								self.conv_encoder_3(
									self.conv_encoder_2(
										self.conv_encoder_1(
											neighbors[:,3:6]
											)
										)
									)], dim=1)              

		# decoder
		inputs = torch.cat([coarse_patches, neighbors_patches], dim=1)
		dec_3 = self.conv_decoder_3(inputs)
		dec_2 = self.conv_decoder_2(dec_3)
		dec_1 = self.conv_decoder_1(dec_2 + coarse_enc_2)
		out = self.conv_decoder_out(dec_1 + coarse_enc_1)

		cnt_end = time()
		return [out], None, None







