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


class SRNRefine(nn.Module):
	def __init__(self, args):
		super(SRNRefine, self).__init__()
		self.n_scales = args.n_scales
		self.args=args

		self.input_layer = nn.Sequential(
				nn.Conv2d(3+3+20 + 14 , 32, 3, stride=1, padding=3//2),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(32 , 32, 3, stride=1, padding=3//2),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(32 , 64, 3, stride=1, padding=3//2),
				nn.LeakyReLU(0.2, inplace=True),
				ResnetBlock(64, 64, 3),
				ResnetBlock(64, 64, 3),
				ResnetBlock(64, 64, 3) 
			)

		self.encoder_1 = nn.Sequential(
				nn.Conv2d(64, 128, 3, stride=2, padding=3//2),
				nn.LeakyReLU(0.2, inplace=True),
				ResnetBlock(128, 128, 3),
				ResnetBlock(128, 128, 3),
				ResnetBlock(128, 128, 3)
			)
		self.encoder_2 = nn.Sequential(
				nn.Conv2d(128, 256, 3, stride=2, padding=3//2),
				nn.LeakyReLU(0.2, inplace=True),
				ResnetBlock(256, 256, 3),
				ResnetBlock(256, 256, 3),
				ResnetBlock(256, 256, 3)
			)

		self.bottle_dilated = nn.Sequential(		# out 256, x0.25
				nn.Conv2d(256, 256, 3, 1, 1, 1),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(256, 256, 3, 1, 2, 2),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(256, 256, 3, 1, 4, 4),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(256, 256, 3, 1, 8, 8),
				nn.LeakyReLU(0.2, inplace=True),
			)

		self.hidden_comb = nn.Sequential(
				nn.Conv2d(512, 256, 3, 1, 1),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(256, 256, 3, 1, 1),
				nn.LeakyReLU(0.2, inplace=True),
			)

		self.decoder_2 = nn.Sequential(
				ResnetBlock(256, 256, 3),
				ResnetBlock(256, 256, 3),
				ResnetBlock(256, 256, 3),
				nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
				nn.LeakyReLU(0.2, inplace=True)
			)
		self.decoder_1 = nn.Sequential(
				ResnetBlock(128, 128, 3),
				ResnetBlock(128, 128, 3),
				ResnetBlock(128, 128, 3),
				nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
				nn.LeakyReLU(0.2, inplace=True)
			)
		self.output_layer = nn.Sequential(
				ResnetBlock(64, 64, 3),
				ResnetBlock(64, 64, 3),
				ResnetBlock(64, 64, 3),
				nn.Conv2d(64, 32, 3, 1, padding=3//2),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(32, 3, 3, 1, padding=3//2)
				# nn.LeakyReLU(0.2, inplace=True)
			)

	def forward(self, input_rgb, input_seg=None, encoded_feat=None):
		# encode neightbor imgs    
		feature=None
		preds = []
		hidden_encon = []
		input_others = torch.cat([input_seg, encoded_feat], dim=1)
		for scale in range(self.n_scales-1, -1, -1):
			scale = 1/(2**scale)
			input_ori = F.interpolate(input_rgb, scale_factor=scale, mode='bilinear', align_corners=True)
			input_pred = F.interpolate(preds[-1].detach(), scale_factor=2, mode='bilinear', align_corners=True) if scale!=1/(2**(self.n_scales-1)) else input_ori
			input_others_scaled = F.interpolate(input_others, scale_factor=scale, mode='bilinear', align_corners=True)			

			input = torch.cat([input_ori, input_pred, input_others_scaled], dim=1)

			input_layer_out = self.input_layer(input)
			encon1_out = self.encoder_1(input_layer_out)
			encon2_out = self.encoder_2(encon1_out)

			bottle_out = self.bottle_dilated(encon2_out)

			last_hidden_encon = F.interpolate(hidden_encon[-1], scale_factor=2, mode='bilinear', align_corners=True) if scale!=1/(2**(self.n_scales-1))\
										else bottle_out

			hidden_encon_input = torch.cat([bottle_out, last_hidden_encon], dim=1)
			decon2_input = self.hidden_comb(hidden_encon_input)
			hidden_encon.append(decon2_input)

			decon2_out = self.decoder_2(decon2_input + encon2_out)
			decon1_out = self.decoder_1(decon2_out + encon1_out)

			pred = self.output_layer(decon1_out + input_layer_out)
			preds.append(pred)

		return preds

########### attention ###############
class MSResAttnRefine(nn.Module):
	def __init__(self, args):
		super(MSResAttnRefine, self).__init__()
		self.args=args
		self.input_layer 		= nn.Sequential(
				nn.Conv2d(3+20, 32, 3, 1, 1),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(32, 64, 3, 1, 1),
				nn.LeakyReLU(0.2, inplace=True)
			)

		self.attn_input_layer 	= nn.Sequential(
				nn.Conv2d(64, 64, 3, 1, 1),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(64, 64, 3, 1, 1),
				nn.LeakyReLU(0.2, inplace=True)
			)
		self.attn_encoder_1 	= nn.Sequential(
				nn.Conv2d(64, 64, 3, 2, 1),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(64, 64, 3, 1, 1),
				nn.LeakyReLU(0.2, inplace=True)
			)
		self.attn_encoder_2 	= nn.Sequential(
				nn.Conv2d(64, 128, 3, 2, 1),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(128, 128, 3, 1, 1),
				nn.LeakyReLU(0.2, inplace=True)
			)
		# self.attn_encoder_3 	= nn.Sequential(
		# 		nn.Conv2d(128, 256, 3, 2, 1),
		# 		nn.LeakyReLU(0.2, inplace=True),
		# 		nn.Conv2d(256, 256, 3, 1, 1),
		# 		nn.LeakyReLU(0.2, inplace=True)
		# 	)
		self.attn_fuse_layer 	= nn.Sequential(
				nn.Conv2d(128, 128, 3, 1, 1),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(128, 128, 3, 1, 1),
				nn.LeakyReLU(0.2, inplace=True),
			)
		self.attn_img_fuse_layer 	= nn.Sequential(
				nn.Conv2d(256, 128, 3, 1, 1),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(128, 128, 3, 1, 1),
				nn.LeakyReLU(0.2, inplace=True),
			) 

		self.img_input_layer 	= nn.Sequential(
				nn.Conv2d(64*3, 64, 3, 1, 1),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(64, 64, 3, 1, 1),
				nn.LeakyReLU(0.2, inplace=True)
			)
		self.img_encoder_1 		= nn.Sequential(
				nn.Conv2d(64, 64, 3, 2, 1),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(64, 64, 3, 1, 1),
				nn.LeakyReLU(0.2, inplace=True)
			)
		self.img_encoder_2 		= nn.Sequential(
				nn.Conv2d(64, 128, 3, 2, 1),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(128, 128, 3, 1, 1),
				nn.LeakyReLU(0.2, inplace=True)
			)
		# self.img_encoder_3 		= nn.Sequential(
		# 		nn.Conv2d(128, 256, 3, 2, 1),
		# 		nn.LeakyReLU(0.2, inplace=True),
		# 		nn.Conv2d(256, 256, 3, 1, 1),
		# 		nn.LeakyReLU(0.2, inplace=True)
		# 	)
		self.img_atrous_layer 	= nn.Sequential(
				nn.Conv2d(128, 128, 3, 1, 1, 1),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(128, 128, 3, 1, 2, 2),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(128, 128, 3, 1, 4, 4),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(128, 128, 3, 1, 8, 8),
				nn.LeakyReLU(0.2, inplace=True)
			)
		self.img_fuse_layer 	= nn.Sequential(
				nn.Conv2d(256, 128, 3, 1, 1),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(128, 128, 3, 1, 1),
				nn.LeakyReLU(0.2, inplace=True),
			) 

		# self.decoder_3 			= nn.Sequential(
		# 		nn.ConvTranspose2d(256, 128, 4, 2, 1),
		# 		nn.LeakyReLU(0.2, inplace=True),
		# 		ResnetBlock(128, 128, 3)
		# 	)
		self.decoder_2 			= nn.Sequential(
				nn.ConvTranspose2d(128, 64, 4, 2, 1),
				nn.LeakyReLU(0.2, inplace=True),
				ResnetBlock(64, 64, 3)
			)
		self.decoder_1 			= nn.Sequential(
				nn.ConvTranspose2d(64, 64, 4, 2, 1),
				nn.LeakyReLU(0.2, inplace=True),
				ResnetBlock(64, 64, 3)
			)
		self.output_layer 		= nn.Sequential(
				nn.Conv2d(64, 64, 3, 1, 1),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(64, 32, 3, 1, 1),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(32, 3, 3, 1, 1)
			)

		self.w = 9
		self.h = 5

	def corrmap(self, x, t1, t2):
		bs, c, H, W = x.size() # e.g. H = 32, W = 64

		x_normed  = x/x.norm(dim=1, keepdim=True)
		t1_normed = t1/t1.norm(dim=1, keepdim=True)
		t2_normed = t2/t2.norm(dim=1, keepdim=True)

		t = torch.cat([t1_normed.unsqueeze(1), t2_normed.unsqueeze(1)], dim=1) # bs, 2, c, H, W
		t_pad = F.pad(t, pad=(self.w//2, self.w//2, self.h//2, self.h//2), value=0)

		t_nns = t_pad.unfold(dimension=3, size=self.h, step=1)# bs, 2, c, H, W, h
		t_nns = t_nns.unfold(dimension=4, size=self.w, step=1)# bs, 2, c, H, W, h, w
		assert t_nns.size()[3:5] == (H, W), t_nns.size()

		x_tf = x_normed.view(bs, 1, c, H, W, 1, 1)

		sim_map = torch.sum(t_nns*x_tf, dim=2) # bs, 2, H, W, h, w
		sim_map_1d = sim_map.view(bs, 2, H, W, self.h*self.w) # bs, 2, H, W, h*w

		# calculate flow_map
		flow_map = sim_map_1d.argmax(dim=4) 
		flow_map = torch.stack([flow_map//self.h, flow_map%self.h], dim=2).float() # bs, 2, 2, H, W

		h_w_add = torch.zeros(bs, 1, 2, H, W)
		h_w_add[:,:,0,:,:] = self.w//2
		h_w_add[:,:,1,:,:] = self.h//2
		flow_map = flow_map.cpu() - h_w_add

		sim_map_1d = torch.cat([sim_map_1d[:,0], sim_map_1d[:, 1]], dim=3) # bs, H, W, 2*h*w
		prob_map = F.softmax(sim_map_1d, dim=3) # bs, H, W, 2*h*w
		if self.args.stage3_prop:
			prob_map = F.avg_pool2d(prob_map.permute(0,3,1,2).contiguous(), 
									kernel_size=(3,5), stride=1, padding=(1,2), count_include_pad=False).permute(0,2,3,1).contiguous()

		return prob_map, flow_map # bs, 2, 2, H, W

	def weight_neighbors_by_low_probmap(self, for_feat, back_feat, prob_map):
		'''
			prob_map: bs, H, W, 2*h*w
			for_feat: bs, c, H, W
		'''
		neighbors_feature = torch.cat([for_feat.unsqueeze(1), back_feat.unsqueeze(1)], dim=1)
		neighbors_feature_pad = F.pad(neighbors_feature, pad=(self.w//2, self.w//2, self.h//2, self.h//2), value=0)
		neighbors_feature_patch = neighbors_feature_pad.unfold(dimension=3, size=self.h, step=1)\
														.unfold(dimension=4, size=self.w, step=1)
		neighbors_feature_patch = neighbors_feature_patch.contiguous().view(list(neighbors_feature_patch.size())[:5] + [self.h*self.w])
														# bs, 2, c, H, W, h*w
		neighbors_feature_patch = torch.cat([neighbors_feature_patch[:,0], neighbors_feature_patch[:,1]], dim=4).contiguous()
		# bs, c, H, W, 2*h*w
		neighbors_feature_weighted = (neighbors_feature_patch*prob_map.unsqueeze(1)).view(list(for_feat.size()) + [2, self.h*self.w])

		# seperately divided by prob_map to reconstruct original scale
		prob_map_split   = prob_map.view(list(prob_map.size())[:3] + [2, self.h*self.w])
		for_denominator  = prob_map_split[:,:,:,0].sum(dim=3)
		back_denominator = prob_map_split[:,:,:,1].sum(dim=3)

		for_feat_weighted  = neighbors_feature_weighted[:,:,:,:,0].sum(dim=4) / for_denominator.unsqueeze(1)
		back_feat_weighted = neighbors_feature_weighted[:,:,:,:,1].sum(dim=4) / back_denominator.unsqueeze(1)
		return 	for_feat_weighted, back_feat_weighted

	def weight_neighbors_by_probmap(self, for_feat, back_feat, prob_map):
		neighbors_feature = torch.cat([for_feat.unsqueeze(1), back_feat.unsqueeze(1)], dim=1)
		neighbors_feature_pad = F.pad(neighbors_feature, pad=(self.w//2, self.w//2, self.h//2, self.h//2), value=0)
		neighbors_feature_patch = neighbors_feature_pad.unfold(dimension=3, size=self.h, step=1)\
														.unfold(dimension=4, size=self.w, step=1)
		neighbors_feature_patch = neighbors_feature_patch.contiguous().view(list(neighbors_feature_patch.size())[:5] + [self.h*self.w])
														# bs, 2, c, H, W, h*w
		neighbors_feature_patch = torch.cat([neighbors_feature_patch[:,0], neighbors_feature_patch[:,1]], dim=4).contiguous()
		# bs, c, H, W, 2*h*w
		neighbors_feature_weighted = (neighbors_feature_patch*prob_map.unsqueeze(1)).sum(dim=4)	
		return 	neighbors_feature_weighted 

	def forward(self, coarse_img, coarse_seg, neighbors_img, neighbors_seg):
		img1 = neighbors_img[:,:3]
		img2 = neighbors_img[:,3:6]
		seg1 = neighbors_seg[:, :20] 
		seg2 = neighbors_seg[:, 20:40] 

		x_comb = torch.cat([coarse_img, coarse_seg], dim=1)
		for_comb = torch.cat([img1, seg1], dim=1)
		back_comb = torch.cat([img2, seg2], dim=1)

		prob_maps = []
		flow_maps = []
		outputs = []
		for scale in range(self.args.n_scales-1, -1, -1):
			scale = 1/(2**scale)
			x_comb_scaled = F.interpolate(x_comb, scale_factor=scale, mode='bilinear', align_corners=True) if scale != 1 else x_comb
			for_comb_scaled = F.interpolate(for_comb, scale_factor=scale, mode='bilinear', align_corners=True) if scale != 1 else for_comb
			back_comb_scaled = F.interpolate(back_comb, scale_factor=scale, mode='bilinear', align_corners=True) if scale != 1 else back_comb
			
			### attn module ####
			x_input_layer_out = self.input_layer(x_comb_scaled)
			x_attn_input_layer_out = self.attn_input_layer(x_input_layer_out)
			x_attn_enc1_out = self.attn_encoder_1(x_attn_input_layer_out)
			x_attn_enc2_out = self.attn_encoder_2(x_attn_enc1_out)
			# x_attn_enc3_out = self.attn_encoder_3(x_attn_enc2_out)

			for_input_layer_out = self.input_layer(for_comb_scaled)
			for_attn_input_layer_out = self.attn_input_layer(for_input_layer_out)
			for_attn_enc1_out = self.attn_encoder_1(for_attn_input_layer_out)
			for_attn_enc2_out = self.attn_encoder_2(for_attn_enc1_out)
			# for_attn_enc3_out = self.attn_encoder_3(for_attn_enc2_out)

			back_input_layer_out = self.input_layer(back_comb_scaled)
			back_attn_input_layer_out = self.attn_input_layer(back_input_layer_out)
			back_attn_enc1_out = self.attn_encoder_1(back_attn_input_layer_out)
			back_attn_enc2_out = self.attn_encoder_2(back_attn_enc1_out)
			# back_attn_enc3_out = self.attn_encoder_3(back_attn_enc2_out)

			# fuse low_scaled prob_map features
			for_attn_enc2_out_weighted, back_attn_enc2_out_weighted = for_attn_enc2_out, back_attn_enc2_out
			if scale != 1/2**(self.args.n_scales-1):
				for k in range(len(prob_maps)):
					low_prob_map = F.interpolate(prob_maps[k].permute(0,3,1,2).contiguous(), scale_factor=2**(len(prob_maps)-k), mode='bilinear', align_corners=True).permute(0,2,3,1).contiguous()
					for_attn_enc2_out_weighted, back_attn_enc2_out_weighted  = self.weight_neighbors_by_low_probmap(
																for_attn_enc2_out_weighted, back_attn_enc2_out_weighted, low_prob_map
															)
				for_attn_enc2_out_weighted = self.attn_fuse_layer(for_attn_enc2_out_weighted)
				back_attn_enc2_out_weighted = self.attn_fuse_layer(back_attn_enc2_out_weighted)

			# calculate residual flow
			prob_map, flow_map = self.corrmap(x_attn_enc2_out, for_attn_enc2_out_weighted, back_attn_enc2_out_weighted)
			prob_maps.append(prob_map)
			flow_maps.append(flow_map)
			# fuse features
			neighbors_feature_weighted = self.weight_neighbors_by_probmap(for_attn_enc2_out, back_attn_enc2_out, prob_map)

			attn_fused_feature = self.attn_img_fuse_layer(torch.cat([x_attn_enc2_out, neighbors_feature_weighted], dim=1))

			### img module ####
			img_input = torch.cat([x_input_layer_out, for_input_layer_out, back_input_layer_out], dim=1) 
			img_input_layer_out = self.img_input_layer(img_input)
			img_enc1_out = self.img_encoder_1(img_input_layer_out)
			img_enc2_out = self.img_encoder_2(img_enc1_out)
			# img_enc3_out = self.img_encoder_3(img_enc2_out)
			img_atrous_out = self.img_atrous_layer(img_enc2_out)

			### fuse and decoder module ###
			fused_feature = self.img_fuse_layer(torch.cat([img_atrous_out, attn_fused_feature], dim=1))
			# dec3_out = self.decoder_3(fused_feature)
			dec2_out = self.decoder_2(fused_feature)
			dec1_out = self.decoder_1(dec2_out + img_enc1_out)
			output = self.output_layer(dec1_out + img_input_layer_out)
			outputs.append(output)

		return outputs, flow_maps


########### attention ###############
class MSResAttnRefineV2(nn.Module):
	def __init__(self, args):
		super(MSResAttnRefineV2, self).__init__()
		self.args=args
		self.input_layer 		= nn.Sequential(
				nn.Conv2d(3+20, 32, 3, 1, 1),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(32, 32, 3, 1, 1),
				nn.LeakyReLU(0.2, inplace=True)
			)
		# encoder layer
		setattr(self, 'encoder_layer_1', 
			nn.Sequential(
					nn.Conv2d(32, 32, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(32, 32, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True)
				)
			)
		setattr(self, 'encoder_layer_2', 
			nn.Sequential(
					nn.Conv2d(32, 64, 3, 2, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(64, 64, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(64, 64, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True),
				)
			)
		setattr(self, 'encoder_layer_3',
			nn.Sequential(
					nn.Conv2d(64, 128, 3, 2, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(128, 128, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(128, 128, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True),
				)
			)
		# downsized layer
		setattr(self, 'attn_down_layer_1', 
			nn.Sequential(
					nn.Conv2d(32, 64, 3, 2, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(64, 128, 3, 2, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(128, 128, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True)
				)
			)
		setattr(self, 'attn_down_layer_2', 
			nn.Sequential(
					nn.Conv2d(64, 128, 3, 2, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(128, 256, 3, 2, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(256, 256, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True)
				)
			)
		setattr(self, 'attn_down_layer_3', 
			nn.Sequential(
					nn.Conv2d(128, 256, 3, 2, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(256, 512, 3, 2, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(512, 512, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True)
				)
			)
		# fuse encoder features with lower level attention flow
		setattr(self, 'layer_1_neighbor_tf', 
			nn.Sequential(
					nn.Conv2d(32, 32, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(32, 32, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True)
				)
			)
		setattr(self, 'layer_2_neighbor_tf',
			nn.Sequential(
					nn.Conv2d(64, 64, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(64, 64, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True)
				)
			)
		# fuse transformed neighbor feature with coarse encoded img
		setattr(self, 'layer_1_fuse', 
				nn.Sequential(
					nn.Conv2d(64, 64, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(64, 32, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True)
					)
				)
		setattr(self, 'layer_2_fuse', 
				nn.Sequential(
					nn.Conv2d(128, 128, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(128, 64, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True)
					)
				)
		setattr(self, 'layer_3_fuse', 
				nn.Sequential(
					nn.Conv2d(256, 256, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(256, 128, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True)
					)
				)
		# decoder layer
		setattr(self, 'decoder_layer_3', 
				nn.Sequential(
					nn.ConvTranspose2d(128, 64, 4, 2, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(64, 64, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(64, 64, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True)
					)
				)
		setattr(self, 'decoder_layer_2',  
			nn.Sequential(
					nn.ConvTranspose2d(64, 32, 4, 2, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(32, 32, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(32, 32, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True)
				)
			)
		# decoded into img directly
		setattr(self, 'out_layer_3', 
				nn.Sequential(
					nn.Conv2d(128, 128, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(128, 64, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(64, 3, 3, 1, 1)
					)
				)
		setattr(self, 'out_layer_2', 
				nn.Sequential(
					nn.Conv2d(64, 64, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(64, 32, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(32, 3, 3, 1, 1)
					)
				)
		setattr(self, 'out_layer_1', 
				nn.Sequential(
					nn.Conv2d(32, 32, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(32, 32, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(32, 3, 3, 1, 1)
					)
				)

		self.w = 5
		self.h = 5

	def corrmap(self, x, t1, t2):
		bs, c, H, W = x.size() # e.g. H = 32, W = 64

		x_normed  = x/x.norm(dim=1, keepdim=True)
		t1_normed = t1/t1.norm(dim=1, keepdim=True)
		t2_normed = t2/t2.norm(dim=1, keepdim=True)

		t = torch.cat([t1_normed.unsqueeze(1), t2_normed.unsqueeze(1)], dim=1) # bs, 2, c, H, W
		t_pad = F.pad(t, pad=(self.w//2, self.w//2, self.h//2, self.h//2), value=0)

		t_nns = t_pad.unfold(dimension=3, size=self.h, step=1)# bs, 2, c, H, W, h
		t_nns = t_nns.unfold(dimension=4, size=self.w, step=1)# bs, 2, c, H, W, h, w
		assert t_nns.size()[3:5] == (H, W), t_nns.size()

		x_tf = x_normed.view(bs, 1, c, H, W, 1, 1)

		sim_map = torch.sum(t_nns*x_tf, dim=2) # bs, 2, H, W, h, w
		sim_map_1d = sim_map.view(bs, 2, H, W, self.h*self.w) # bs, 2, H, W, h*w

		# calculate flow_map
		flow_map = sim_map_1d.argmax(dim=4) 
		flow_map = torch.stack([flow_map//self.h, flow_map%self.h], dim=2).float() # bs, 2, 2, H, W

		h_w_add = torch.zeros(bs, 1, 2, H, W)
		h_w_add[:,:,0,:,:] = self.w//2
		h_w_add[:,:,1,:,:] = self.h//2
		flow_map = flow_map.cpu() - h_w_add

		sim_map_1d = torch.cat([sim_map_1d[:,0], sim_map_1d[:, 1]], dim=3) # bs, H, W, 2*h*w
		sim_map_1d = F.interpolate(sim_map_1d.permute(0,3,1,2).contiguous(), scale_factor=4, mode='bilinear', align_corners=True)\
						.permute(0,2,3,1).contiguous()
		prob_map = F.softmax(sim_map_1d, dim=3) # bs, H, W, 2*h*w
		if self.args.stage3_prop:
			prob_map = F.avg_pool2d(prob_map.permute(0,3,1,2).contiguous(), 
									kernel_size=(3,3), stride=1, padding=(1,1), count_include_pad=False).permute(0,2,3,1).contiguous()
		return prob_map, flow_map # bs, 2, 2, H, W

	def weight_neighbors_by_low_probmap(self, for_feat, back_feat, prob_map):
		'''
			prob_map: bs, H, W, 2*h*w
			for_feat: bs, c, H, W
		'''
		neighbors_feature = torch.cat([for_feat.unsqueeze(1), back_feat.unsqueeze(1)], dim=1)
		neighbors_feature_pad = F.pad(neighbors_feature, pad=(self.w//2, self.w//2, self.h//2, self.h//2), value=0)
		neighbors_feature_patch = neighbors_feature_pad.unfold(dimension=3, size=self.h, step=1)\
														.unfold(dimension=4, size=self.w, step=1)
		neighbors_feature_patch = neighbors_feature_patch.contiguous().view(list(neighbors_feature_patch.size())[:5] + [self.h*self.w])
														# bs, 2, c, H, W, h*w
		neighbors_feature_patch = torch.cat([neighbors_feature_patch[:,0], neighbors_feature_patch[:,1]], dim=4).contiguous()
		# bs, c, H, W, 2*h*w
		neighbors_feature_weighted = (neighbors_feature_patch*prob_map.unsqueeze(1)).view(list(for_feat.size()) + [2, self.h*self.w])

		# seperately divided by prob_map to reconstruct original scale
		prob_map_split   = prob_map.view(list(prob_map.size())[:3] + [2, self.h*self.w])
		for_denominator  = prob_map_split[:,:,:,0].sum(dim=3)
		back_denominator = prob_map_split[:,:,:,1].sum(dim=3)

		for_feat_weighted  = neighbors_feature_weighted[:,:,:,:,0].sum(dim=4) / for_denominator.unsqueeze(1)
		back_feat_weighted = neighbors_feature_weighted[:,:,:,:,1].sum(dim=4) / back_denominator.unsqueeze(1)
		return 	for_feat_weighted, back_feat_weighted

	def weight_neighbors_by_probmap(self, for_feat, back_feat, prob_map):
		neighbors_feature = torch.cat([for_feat.unsqueeze(1), back_feat.unsqueeze(1)], dim=1)
		neighbors_feature_pad = F.pad(neighbors_feature, pad=(self.w//2, self.w//2, self.h//2, self.h//2), value=0)
		neighbors_feature_patch = neighbors_feature_pad.unfold(dimension=3, size=self.h, step=1)\
														.unfold(dimension=4, size=self.w, step=1)
		neighbors_feature_patch = neighbors_feature_patch.contiguous().view(list(neighbors_feature_patch.size())[:5] + [self.h*self.w])
														# bs, 2, c, H, W, h*w
		neighbors_feature_patch = torch.cat([neighbors_feature_patch[:,0], neighbors_feature_patch[:,1]], dim=4).contiguous()
		# bs, c, H, W, 2*h*w
		neighbors_feature_weighted = (neighbors_feature_patch*prob_map.unsqueeze(1)).sum(dim=4)	
		return 	neighbors_feature_weighted 

	def forward(self, coarse_img, coarse_seg, neighbors_img, neighbors_seg):
		img1 = neighbors_img[:,:3]
		img2 = neighbors_img[:,3:6]
		seg1 = neighbors_seg[:, :20] 
		seg2 = neighbors_seg[:, 20:40] 

		x_comb = torch.cat([coarse_img, coarse_seg], dim=1)
		for_comb = torch.cat([img1, seg1], dim=1)
		back_comb = torch.cat([img2, seg2], dim=1)

		prob_maps = []
		flow_maps = []
		outputs = []

		### attn module ####
		x_input_layer_out = self.input_layer(x_comb)
		x_attn_enc_layer_1_out = self.encoder_layer_1(x_input_layer_out)
		x_attn_enc_layer_2_out = self.encoder_layer_2(x_attn_enc_layer_1_out)
		x_attn_enc_layer_3_out = self.encoder_layer_3(x_attn_enc_layer_2_out)
		x_attn_enc_features = [x_attn_enc_layer_3_out, x_attn_enc_layer_2_out, x_attn_enc_layer_1_out]

		for_input_layer_out = self.input_layer(for_comb)
		for_attn_enc_layer_1_out = self.encoder_layer_1(for_input_layer_out)
		for_attn_enc_layer_2_out = self.encoder_layer_2(for_attn_enc_layer_1_out)
		for_attn_enc_layer_3_out = self.encoder_layer_3(for_attn_enc_layer_2_out)
		for_attn_enc_features = [for_attn_enc_layer_3_out, for_attn_enc_layer_2_out, for_attn_enc_layer_1_out]

		back_input_layer_out = self.input_layer(back_comb)
		back_attn_enc_layer_1_out = self.encoder_layer_1(back_input_layer_out)
		back_attn_enc_layer_2_out = self.encoder_layer_2(back_attn_enc_layer_1_out)
		back_attn_enc_layer_3_out = self.encoder_layer_3(back_attn_enc_layer_2_out)
		back_attn_enc_features = [back_attn_enc_layer_3_out, back_attn_enc_layer_2_out, back_attn_enc_layer_1_out]

		fused_features = []
		for i in range(3):
			for k in range(len(prob_maps)):
				# transform lowest level feature
				low_prob_map = F.interpolate(prob_maps[k].permute(0,3,1,2).contiguous(), scale_factor=2**(len(prob_maps)-k), mode='bilinear', align_corners=True).permute(0,2,3,1).contiguous()
				for_attn_enc_features[i], back_attn_enc_features[i]  = self.weight_neighbors_by_low_probmap(
															for_attn_enc_features[i], back_attn_enc_features[i], low_prob_map
														)
			if i!= 0:
				for_attn_enc_features[i] = getattr(self, 'layer_{}_neighbor_tf'.format(3-i))(for_attn_enc_features[i])
				back_attn_enc_features[i] = getattr(self, 'layer_{}_neighbor_tf'.format(3-i))(back_attn_enc_features[i])

			# calculate residual flow
			down_x = getattr(self, 'attn_down_layer_{}'.format(3-i))(x_attn_enc_features[i])
			down_for = getattr(self, 'attn_down_layer_{}'.format(3-i))(for_attn_enc_features[i])
			down_back = getattr(self, 'attn_down_layer_{}'.format(3-i))(back_attn_enc_features[i])
			prob_map, flow_map = self.corrmap(down_x, down_for, down_back)
			prob_maps.append(prob_map)
			flow_maps.append(flow_map)
			# fuse features
			neighbors_feature_weighted = self.weight_neighbors_by_probmap(for_attn_enc_features[i], back_attn_enc_features[i], prob_map)

			fused_feature = getattr(self, 'layer_{}_fuse'.format(3-i))(torch.cat([x_attn_enc_features[i], neighbors_feature_weighted], dim=1))
			if i!=0:
				fused_feature = fused_feature + fused_features[-1]
			if i!=2:
				fused_features.append(getattr(self, 'decoder_layer_{}'.format(3-i))(fused_feature))
			output = getattr(self, 'out_layer_{}'.format(3-i))(fused_feature)
			outputs.append(output)

		return outputs, flow_maps



########### attention ###############
class MSResAttnRefineV2Base(nn.Module):
	def __init__(self, args):
		super(MSResAttnRefineV2Base, self).__init__()
		self.args=args
		self.input_layer 		= nn.Sequential(
				nn.Conv2d(3+20, 32, 3, 1, 1),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(32, 32, 3, 1, 1),
				nn.LeakyReLU(0.2, inplace=True)
			)
		# encoder layer
		setattr(self, 'encoder_layer_1', 
			nn.Sequential(
					nn.Conv2d(32, 32, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(32, 32, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True)
				)
			)
		setattr(self, 'encoder_layer_2', 
			nn.Sequential(
					nn.Conv2d(32, 64, 3, 2, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(64, 64, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(64, 64, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True),
				)
			)
		setattr(self, 'encoder_layer_3',
			nn.Sequential(
					nn.Conv2d(64, 128, 3, 2, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(128, 128, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(128, 128, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True),
				)
			)
		# fuse encoder features with lower level attention flow
		setattr(self, 'layer_1_neighbor_tf', 
			nn.Sequential(
					nn.Conv2d(32, 32, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(32, 32, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True)
				)
			)
		setattr(self, 'layer_2_neighbor_tf',
			nn.Sequential(
					nn.Conv2d(64, 64, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(64, 64, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True)
				)
			)
		# fuse transformed neighbor feature with coarse encoded img
		setattr(self, 'layer_1_fuse', 
				nn.Sequential(
					nn.Conv2d(64, 64, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(64, 32, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True)
					)
				)
		setattr(self, 'layer_2_fuse', 
				nn.Sequential(
					nn.Conv2d(128, 128, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(128, 64, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True)
					)
				)
		setattr(self, 'layer_3_fuse', 
				nn.Sequential(
					nn.Conv2d(256, 256, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(256, 128, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True)
					)
				)
		# decoder layer
		setattr(self, 'decoder_layer_3', 
				nn.Sequential(
					nn.ConvTranspose2d(128, 64, 4, 2, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(64, 64, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(64, 64, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True)
					)
				)
		setattr(self, 'decoder_layer_2',  
			nn.Sequential(
					nn.ConvTranspose2d(64, 32, 4, 2, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(32, 32, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(32, 32, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True)
				)
			)
		# decoded into img directly
		setattr(self, 'out_layer_3', 
				nn.Sequential(
					nn.Conv2d(128, 128, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(128, 64, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(64, 3, 3, 1, 1)
					)
				)
		setattr(self, 'out_layer_2', 
				nn.Sequential(
					nn.Conv2d(64, 64, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(64, 32, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(32, 3, 3, 1, 1)
					)
				)
		setattr(self, 'out_layer_1', 
				nn.Sequential(
					nn.Conv2d(32, 32, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(32, 32, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(32, 3, 3, 1, 1)
					)
				)

		self.w = 5
		self.h = 5

	def corrmap(self, x, t1, t2):
		bs, c, H, W = x.size() # e.g. H = 32, W = 64

		x_normed  = x/x.norm(dim=1, keepdim=True)
		t1_normed = t1/t1.norm(dim=1, keepdim=True)
		t2_normed = t2/t2.norm(dim=1, keepdim=True)

		t = torch.cat([t1_normed.unsqueeze(1), t2_normed.unsqueeze(1)], dim=1) # bs, 2, c, H, W
		t_pad = F.pad(t, pad=(self.w//2, self.w//2, self.h//2, self.h//2), value=0)

		t_nns = t_pad.unfold(dimension=3, size=self.h, step=1)# bs, 2, c, H, W, h
		t_nns = t_nns.unfold(dimension=4, size=self.w, step=1)# bs, 2, c, H, W, h, w
		assert t_nns.size()[3:5] == (H, W), t_nns.size()

		x_tf = x_normed.view(bs, 1, c, H, W, 1, 1)

		sim_map = torch.sum(t_nns*x_tf, dim=2) # bs, 2, H, W, h, w
		sim_map_1d = sim_map.view(bs, 2, H, W, self.h*self.w) # bs, 2, H, W, h*w

		# calculate flow_map
		flow_map = sim_map_1d.argmax(dim=4) 
		flow_map = torch.stack([flow_map//self.h, flow_map%self.h], dim=2).float() # bs, 2, 2, H, W

		h_w_add = torch.zeros(bs, 1, 2, H, W)
		h_w_add[:,:,0,:,:] = self.w//2
		h_w_add[:,:,1,:,:] = self.h//2
		flow_map = flow_map.cpu() - h_w_add

		sim_map_1d = torch.cat([sim_map_1d[:,0], sim_map_1d[:, 1]], dim=3) # bs, H, W, 2*h*w
		sim_map_1d = F.interpolate(sim_map_1d.permute(0,3,1,2).contiguous(), scale_factor=4, mode='bilinear', align_corners=True)\
						.permute(0,2,3,1).contiguous()
		prob_map = F.softmax(sim_map_1d, dim=3) # bs, H, W, 2*h*w
		if self.args.stage3_prop:
			prob_map = F.avg_pool2d(prob_map.permute(0,3,1,2).contiguous(), 
									kernel_size=(3,3), stride=1, padding=(1,1), count_include_pad=False).permute(0,2,3,1).contiguous()
		return prob_map, flow_map # bs, 2, 2, H, W

	def forward(self, coarse_img, coarse_seg, neighbors_img, neighbors_seg):
		img1 = neighbors_img[:,:3]
		img2 = neighbors_img[:,3:6]
		seg1 = neighbors_seg[:, :20] 
		seg2 = neighbors_seg[:, 20:40] 

		x_comb = torch.cat([coarse_img, coarse_seg], dim=1)
		for_comb = torch.cat([img1, seg1], dim=1)
		back_comb = torch.cat([img2, seg2], dim=1)

		prob_maps = []
		flow_maps = []
		outputs = []

		### attn module ####
		x_input_layer_out = self.input_layer(x_comb)
		x_attn_enc_layer_1_out = self.encoder_layer_1(x_input_layer_out)
		x_attn_enc_layer_2_out = self.encoder_layer_2(x_attn_enc_layer_1_out)
		x_attn_enc_layer_3_out = self.encoder_layer_3(x_attn_enc_layer_2_out)
		x_attn_enc_features = [x_attn_enc_layer_3_out, x_attn_enc_layer_2_out, x_attn_enc_layer_1_out]

		for_input_layer_out = self.input_layer(for_comb)
		for_attn_enc_layer_1_out = self.encoder_layer_1(for_input_layer_out)
		for_attn_enc_layer_2_out = self.encoder_layer_2(for_attn_enc_layer_1_out)
		for_attn_enc_layer_3_out = self.encoder_layer_3(for_attn_enc_layer_2_out)
		for_attn_enc_features = [for_attn_enc_layer_3_out, for_attn_enc_layer_2_out, for_attn_enc_layer_1_out]

		back_input_layer_out = self.input_layer(back_comb)
		back_attn_enc_layer_1_out = self.encoder_layer_1(back_input_layer_out)
		back_attn_enc_layer_2_out = self.encoder_layer_2(back_attn_enc_layer_1_out)
		back_attn_enc_layer_3_out = self.encoder_layer_3(back_attn_enc_layer_2_out)
		back_attn_enc_features = [back_attn_enc_layer_3_out, back_attn_enc_layer_2_out, back_attn_enc_layer_1_out]

		fused_features = []
		for i in range(3):
			if i!= 0:
				for_attn_enc_features[i] = getattr(self, 'layer_{}_neighbor_tf'.format(3-i))(for_attn_enc_features[i])
				back_attn_enc_features[i] = getattr(self, 'layer_{}_neighbor_tf'.format(3-i))(back_attn_enc_features[i])

			# fuse features
			neighbors_feature_weighted = for_attn_enc_features[i] + back_attn_enc_features[i]

			fused_feature = getattr(self, 'layer_{}_fuse'.format(3-i))(torch.cat([x_attn_enc_features[i], neighbors_feature_weighted], dim=1))
			if i!=0:
				fused_feature = fused_feature + fused_features[-1]
			if i!=2:
				fused_features.append(getattr(self, 'decoder_layer_{}'.format(3-i))(fused_feature))
			output = getattr(self, 'out_layer_{}'.format(3-i))(fused_feature)
			outputs.append(output)

		return outputs, None


########### attention ###############
class MSResAttnRefineV3(nn.Module):
	def __init__(self, args):
		super(MSResAttnRefineV3, self).__init__()
		self.args=args
		self.input_layer 		= nn.Sequential(
				nn.Conv2d(3+20, 32, 3, 1, 1),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(32, 32, 3, 1, 1),
				nn.LeakyReLU(0.2, inplace=True),
				ResnetBlock(32, 32, 3),
				ResnetBlock(32, 32, 3)
			)
		# encoder layer
		setattr(self, 'encoder_layer_1', 
			nn.Sequential(
					nn.Conv2d(32, 32, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True),
					ResnetBlock(32,32,3),
					ResnetBlock(32,32,3)
				)
			)
		setattr(self, 'encoder_layer_2', 
			nn.Sequential(
					nn.Conv2d(32, 64, 3, 2, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(64, 64, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True),
					ResnetBlock(64,64,3),
					ResnetBlock(64,64,3)
				)
			)
		setattr(self, 'encoder_layer_3',
			nn.Sequential(
					nn.Conv2d(64, 128, 3, 2, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(128, 128, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True),
					ResnetBlock(128,128,3),
					ResnetBlock(128,128,3)
				)
			)
		# downsized layer
		setattr(self, 'attn_down_layer_1', 
			nn.Sequential(
					nn.Conv2d(32, 64, 3, 2, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(64, 128, 3, 2, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(128, 128, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True)
				)
			)
		setattr(self, 'attn_down_layer_2', 
			nn.Sequential(
					nn.Conv2d(64, 128, 3, 2, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(128, 256, 3, 2, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(256, 256, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True)
				)
			)
		setattr(self, 'attn_down_layer_3', 
			nn.Sequential(
					nn.Conv2d(128, 256, 3, 2, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(256, 512, 3, 2, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(512, 512, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True)
				)
			)
		# fuse encoder features with lower level attention flow
		setattr(self, 'layer_1_neighbor_tf', 
			nn.Sequential(
					nn.Conv2d(32, 32, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(32, 32, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True),
					ResnetBlock(32,32,3),
					ResnetBlock(32,32,3)
				)
			)
		setattr(self, 'layer_2_neighbor_tf',
			nn.Sequential(
					nn.Conv2d(64, 64, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(64, 64, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True),
					ResnetBlock(64, 64, 3),
					ResnetBlock(64, 64, 3)
				)
			)
		# fuse transformed neighbor feature with coarse encoded img
		# setattr(self, 'layer_1_fuse', 
		# 		nn.Sequential(
		# 			nn.Conv2d(64, 64, 3, 1, 1),
		# 			nn.LeakyReLU(0.2, inplace=True),
		# 			nn.Conv2d(64, 32, 3, 1, 1),
		# 			nn.LeakyReLU(0.2, inplace=True)
		# 			)
		# 		)
		# setattr(self, 'layer_2_fuse', 
		# 		nn.Sequential(
		# 			nn.Conv2d(128, 128, 3, 1, 1),
		# 			nn.LeakyReLU(0.2, inplace=True),
		# 			nn.Conv2d(128, 64, 3, 1, 1),
		# 			nn.LeakyReLU(0.2, inplace=True)
		# 			)
		# 		)
		setattr(self, 'layer_3_fuse', 
				nn.Sequential(
					nn.Conv2d(256, 256, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(256, 128, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True),
					ResnetBlock(128,128,3),
					ResnetBlock(128,128,3)
					)
				)
		# decoder layer
		setattr(self, 'decoder_layer_3', 
				nn.Sequential(
					nn.ConvTranspose2d(128, 64, 4, 2, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(64, 64, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True),
					ResnetBlock(64, 64, 3),
					ResnetBlock(64, 64, 3)
					)
				)
		setattr(self, 'decoder_layer_2',  
			nn.Sequential(
					nn.ConvTranspose2d(64, 32, 4, 2, 1),
					nn.LeakyReLU(0.2, inplace=True),
					nn.Conv2d(32, 32, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True),
					ResnetBlock(32, 32, 3),
					ResnetBlock(32, 32, 3)
				)
			)
		# decoded into img directly
		setattr(self, 'out_layer_3', 
				nn.Sequential(
					nn.Conv2d(128, 64, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True),
					ResnetBlock(64, 64, 3),
					ResnetBlock(64, 64, 3),
					nn.Conv2d(64, 3, 3, 1, 1)
					)
				)
		setattr(self, 'out_layer_2', 
				nn.Sequential(
					nn.Conv2d(64, 32, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True),
					ResnetBlock(32, 32, 3),
					ResnetBlock(32, 32, 3),
					nn.Conv2d(32, 3, 3, 1, 1)
					)
				)
		setattr(self, 'out_layer_1', 
				nn.Sequential(
					nn.Conv2d(32, 32, 3, 1, 1),
					nn.LeakyReLU(0.2, inplace=True),
					ResnetBlock(32, 32, 3),
					ResnetBlock(32, 32, 3),
					nn.Conv2d(32, 3, 3, 1, 1)
					)
				)

		self.w = 5
		self.h = 5

	def corrmap(self, x, t1, t2):
		bs, c, H, W = x.size() # e.g. H = 32, W = 64

		x_normed  = x/x.norm(dim=1, keepdim=True)
		t1_normed = t1/t1.norm(dim=1, keepdim=True)
		t2_normed = t2/t2.norm(dim=1, keepdim=True)

		t = torch.cat([t1_normed.unsqueeze(1), t2_normed.unsqueeze(1)], dim=1) # bs, 2, c, H, W
		t_pad = F.pad(t, pad=(self.w//2, self.w//2, self.h//2, self.h//2), value=0)

		t_nns = t_pad.unfold(dimension=3, size=self.h, step=1)# bs, 2, c, H, W, h
		t_nns = t_nns.unfold(dimension=4, size=self.w, step=1)# bs, 2, c, H, W, h, w
		assert t_nns.size()[3:5] == (H, W), t_nns.size()

		x_tf = x_normed.view(bs, 1, c, H, W, 1, 1)

		sim_map = torch.sum(t_nns*x_tf, dim=2) # bs, 2, H, W, h, w
		sim_map_1d = sim_map.view(bs, 2, H, W, self.h*self.w) # bs, 2, H, W, h*w

		# calculate flow_map
		flow_map = sim_map_1d.argmax(dim=4) 
		flow_map = torch.stack([flow_map//self.h, flow_map%self.h], dim=2).float() # bs, 2, 2, H, W

		h_w_add = torch.zeros(bs, 1, 2, H, W)
		h_w_add[:,:,0,:,:] = self.w//2
		h_w_add[:,:,1,:,:] = self.h//2
		flow_map = flow_map.cpu() - h_w_add

		sim_map_1d = torch.cat([sim_map_1d[:,0], sim_map_1d[:, 1]], dim=3) # bs, H, W, 2*h*w
		sim_map_1d = F.interpolate(sim_map_1d.permute(0,3,1,2).contiguous(), scale_factor=4, mode='bilinear', align_corners=True)\
						.permute(0,2,3,1).contiguous()
		prob_map = F.softmax(sim_map_1d, dim=3) # bs, H, W, 2*h*w
		if self.args.stage3_prop:
			prob_map = F.avg_pool2d(prob_map.permute(0,3,1,2).contiguous(), 
									kernel_size=(3,3), stride=1, padding=(1,1), count_include_pad=False).permute(0,2,3,1).contiguous()
		return prob_map, flow_map # bs, 2, 2, H, W

	def weight_neighbors_by_low_probmap(self, for_feat, back_feat, prob_map):
		'''
			prob_map: bs, H, W, 2*h*w
			for_feat: bs, c, H, W
		'''
		neighbors_feature = torch.cat([for_feat.unsqueeze(1), back_feat.unsqueeze(1)], dim=1)
		neighbors_feature_pad = F.pad(neighbors_feature, pad=(self.w//2, self.w//2, self.h//2, self.h//2), value=0)
		neighbors_feature_patch = neighbors_feature_pad.unfold(dimension=3, size=self.h, step=1)\
														.unfold(dimension=4, size=self.w, step=1)
		neighbors_feature_patch = neighbors_feature_patch.contiguous().view(list(neighbors_feature_patch.size())[:5] + [self.h*self.w])
														# bs, 2, c, H, W, h*w
		neighbors_feature_patch = torch.cat([neighbors_feature_patch[:,0], neighbors_feature_patch[:,1]], dim=4).contiguous()
		# bs, c, H, W, 2*h*w
		neighbors_feature_weighted = (neighbors_feature_patch*prob_map.unsqueeze(1)).view(list(for_feat.size()) + [2, self.h*self.w])

		# seperately divided by prob_map to reconstruct original scale
		prob_map_split   = prob_map.view(list(prob_map.size())[:3] + [2, self.h*self.w])
		for_denominator  = prob_map_split[:,:,:,0].sum(dim=3)
		back_denominator = prob_map_split[:,:,:,1].sum(dim=3)

		for_feat_weighted  = neighbors_feature_weighted[:,:,:,:,0].sum(dim=4) / for_denominator.unsqueeze(1)
		back_feat_weighted = neighbors_feature_weighted[:,:,:,:,1].sum(dim=4) / back_denominator.unsqueeze(1)
		return 	for_feat_weighted, back_feat_weighted

	def weight_neighbors_by_probmap(self, for_feat, back_feat, prob_map):
		neighbors_feature = torch.cat([for_feat.unsqueeze(1), back_feat.unsqueeze(1)], dim=1)
		neighbors_feature_pad = F.pad(neighbors_feature, pad=(self.w//2, self.w//2, self.h//2, self.h//2), value=0)
		neighbors_feature_patch = neighbors_feature_pad.unfold(dimension=3, size=self.h, step=1)\
														.unfold(dimension=4, size=self.w, step=1)
		neighbors_feature_patch = neighbors_feature_patch.contiguous().view(list(neighbors_feature_patch.size())[:5] + [self.h*self.w])
														# bs, 2, c, H, W, h*w
		neighbors_feature_patch = torch.cat([neighbors_feature_patch[:,0], neighbors_feature_patch[:,1]], dim=4).contiguous()
		# bs, c, H, W, 2*h*w
		neighbors_feature_weighted = (neighbors_feature_patch*prob_map.unsqueeze(1)).sum(dim=4)	
		return 	neighbors_feature_weighted 

	def forward(self, coarse_img, coarse_seg, neighbors_img, neighbors_seg):
		img1 = neighbors_img[:,:3]
		img2 = neighbors_img[:,3:6]
		seg1 = neighbors_seg[:, :20] 
		seg2 = neighbors_seg[:, 20:40] 

		x_comb = torch.cat([coarse_img, coarse_seg], dim=1)
		for_comb = torch.cat([img1, seg1], dim=1)
		back_comb = torch.cat([img2, seg2], dim=1)

		prob_maps = []
		flow_maps = []
		outputs = []

		### attn module ####
		x_input_layer_out = self.input_layer(x_comb)
		x_attn_enc_layer_1_out = self.encoder_layer_1(x_input_layer_out)
		x_attn_enc_layer_2_out = self.encoder_layer_2(x_attn_enc_layer_1_out)
		x_attn_enc_layer_3_out = self.encoder_layer_3(x_attn_enc_layer_2_out)
		x_attn_enc_features = [x_attn_enc_layer_3_out, x_attn_enc_layer_2_out, x_attn_enc_layer_1_out]

		for_input_layer_out = self.input_layer(for_comb)
		for_attn_enc_layer_1_out = self.encoder_layer_1(for_input_layer_out)
		for_attn_enc_layer_2_out = self.encoder_layer_2(for_attn_enc_layer_1_out)
		for_attn_enc_layer_3_out = self.encoder_layer_3(for_attn_enc_layer_2_out)
		for_attn_enc_features = [for_attn_enc_layer_3_out, for_attn_enc_layer_2_out, for_attn_enc_layer_1_out]

		back_input_layer_out = self.input_layer(back_comb)
		back_attn_enc_layer_1_out = self.encoder_layer_1(back_input_layer_out)
		back_attn_enc_layer_2_out = self.encoder_layer_2(back_attn_enc_layer_1_out)
		back_attn_enc_layer_3_out = self.encoder_layer_3(back_attn_enc_layer_2_out)
		back_attn_enc_features = [back_attn_enc_layer_3_out, back_attn_enc_layer_2_out, back_attn_enc_layer_1_out]

		fused_features = []
		for i in range(3):
			for k in range(len(prob_maps)):
				# transform lowest level feature
				low_prob_map = F.interpolate(prob_maps[k].permute(0,3,1,2).contiguous(), scale_factor=2**(len(prob_maps)-k), mode='bilinear', align_corners=True).permute(0,2,3,1).contiguous()
				for_attn_enc_features[i], back_attn_enc_features[i]  = self.weight_neighbors_by_low_probmap(
															for_attn_enc_features[i], back_attn_enc_features[i], low_prob_map
														)
			if i!= 0:
				for_attn_enc_features[i] = getattr(self, 'layer_{}_neighbor_tf'.format(3-i))(for_attn_enc_features[i])
				back_attn_enc_features[i] = getattr(self, 'layer_{}_neighbor_tf'.format(3-i))(back_attn_enc_features[i])

			# calculate residual flow
			down_x = getattr(self, 'attn_down_layer_{}'.format(3-i))(x_attn_enc_features[i])
			down_for = getattr(self, 'attn_down_layer_{}'.format(3-i))(for_attn_enc_features[i])
			down_back = getattr(self, 'attn_down_layer_{}'.format(3-i))(back_attn_enc_features[i])
			prob_map, flow_map = self.corrmap(down_x, down_for, down_back)
			prob_maps.append(prob_map)
			flow_maps.append(flow_map)
			# fuse features
			neighbors_feature_weighted = self.weight_neighbors_by_probmap(for_attn_enc_features[i], back_attn_enc_features[i], prob_map)

			if i == 0:
				fused_feature = getattr(self, 'layer_{}_fuse'.format(3-i))(torch.cat([x_attn_enc_features[i], neighbors_feature_weighted], dim=1))
			else:
				fused_feature = neighbors_feature_weighted
			if i!=0:
				fused_feature = fused_feature + fused_features[-1]
			if i!=2:
				fused_features.append(getattr(self, 'decoder_layer_{}'.format(3-i))(fused_feature))
			output = getattr(self, 'out_layer_{}'.format(3-i))(fused_feature)
			outputs.append(output)

		return outputs, flow_maps



