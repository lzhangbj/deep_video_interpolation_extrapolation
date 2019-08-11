import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import random
import numpy as np

class TrackGen(nn.Module):
	def __init__(self, args):
		super(TrackGen, self).__init__()
		self.args = args

		self.encoder_1 = nn.Sequential(
				nn.Conv2d(46, 32, 3, 1, 1),
				nn.LeakyReLU(0.2,inplace=True),
				nn.Conv2d(32, 32, 3, 1, 1),
				nn.LeakyReLU(0.2,inplace=True),
				nn.Conv2d(32, 32, 3, 1, 1),
				nn.LeakyReLU(0.2,inplace=True)			# 64x64
			)
		self.encoder_2 = nn.Sequential(
				nn.Conv2d(32, 64, 3, 2, 1),
				nn.LeakyReLU(0.2,inplace=True),
				nn.Conv2d(64, 64, 3, 1, 1),
				nn.LeakyReLU(0.2,inplace=True),
				nn.Conv2d(64, 64, 3, 1, 1),
				nn.LeakyReLU(0.2,inplace=True)			# 32x32
			)
		self.encoder_3 = nn.Sequential(
				nn.Conv2d(64, 128, 3, 2, 1),
				nn.LeakyReLU(0.2,inplace=True),
				nn.Conv2d(128, 128, 3, 1, 1),
				nn.LeakyReLU(0.2,inplace=True),
				nn.Conv2d(128, 128, 3, 1, 1),
				nn.LeakyReLU(0.2,inplace=True)			# 16x16
			)
		self.encoder_4 = nn.Sequential(
				nn.Conv2d(128, 128, 3, 2, 1),
				nn.LeakyReLU(0.2,inplace=True),
				nn.Conv2d(128, 128, 3, 1, 1),
				nn.LeakyReLU(0.2,inplace=True),
				nn.Conv2d(128, 128, 3, 1, 1),
				nn.LeakyReLU(0.2,inplace=True)			# 8x8
			)

		self.bottom_layer = nn.Sequential(
				nn.Conv2d(128, 256, 3, 2, 1),
				nn.LeakyReLU(0.2,inplace=True),
				nn.Conv2d(256, 256, 3, 1, 1),
				nn.LeakyReLU(0.2,inplace=True),
				nn.Conv2d(256, 256, 3, 1, 1),
				nn.LeakyReLU(0.2,inplace=True)			# 4x4
			)

		self.up_4 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True))
		self.decoder_4 = nn.Sequential(
				nn.Conv2d(128*2, 128, 3, 1, 1),
				nn.LeakyReLU(0.2,inplace=True),
				nn.Conv2d(128, 128, 3, 1, 1),
				nn.LeakyReLU(0.2,inplace=True),
				nn.Conv2d(128, 128, 3, 1, 1),
				nn.LeakyReLU(0.2,inplace=True)			# 8x8
			)
		self.up_3 = nn.Sequential(nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True))
		self.decoder_3 = nn.Sequential(
				nn.Conv2d(128*2, 128, 3, 1, 1),
				nn.LeakyReLU(0.2,inplace=True),
				nn.Conv2d(128, 128, 3, 1, 1),
				nn.LeakyReLU(0.2,inplace=True),
				nn.Conv2d(128, 128, 3, 1, 1),
				nn.LeakyReLU(0.2,inplace=True)			# 16x16
			)
		self.up_2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True))
		self.decoder_2 = nn.Sequential(
				nn.Conv2d(64*2, 64, 3, 1, 1),
				nn.LeakyReLU(0.2,inplace=True),
				nn.Conv2d(64, 64, 3, 1, 1),
				nn.LeakyReLU(0.2,inplace=True),
				nn.Conv2d(64, 64, 3, 1, 1),
				nn.LeakyReLU(0.2,inplace=True)			# 32x32
			)
		self.up_1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True))
		self.decoder_1 = nn.Sequential(
				nn.Conv2d(32*2, 32, 3, 1, 1),
				nn.LeakyReLU(0.2,inplace=True),
				nn.Conv2d(32, 32, 3, 1, 1),
				nn.LeakyReLU(0.2,inplace=True),
				nn.Conv2d(32, 32, 3, 1, 1),
				nn.LeakyReLU(0.2,inplace=True)			# 64x64
			)

		TRACK_NUM = self.args.num_track_per_img

		self.track_fusion_layer = nn.Sequential(
				nn.Conv2d(32*TRACK_NUM, 128, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(128, 64, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(64, 32, 3, 1, 1)
			)

		self.fusion_layer = nn.Sequential(
				nn.Conv2d(32+3+20+1, 48, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(48, 48, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(48, 32, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(32, 32, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(32, 32, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True)
			)

		self.rgb_out_layer = nn.Sequential(
				nn.Conv2d(32, 32, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(32, 3, 3, 1, 1)
			)
		self.seg_out_layer = nn.Sequential(
				nn.Conv2d(32, 32, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(32, 20, 3, 1, 1)
			)

		self.H = 64
		self.W = 64

		self.track_num = self.args.num_track_per_img

	def forward(self, input, coarse_rgb, coarse_seg, bboxes):
		bs = input.size(0)
		TRACK_NUM = self.track_num
		for_img  = torch.cat([input[:,:3], input[:,6:26]], dim=1)
		back_img = torch.cat([input[:,3:6], input[:,26:46]], dim=1)

		track_patches = []
		for i in range(bs):
			for j in range(TRACK_NUM):
				# mid_box = bboxes[i, 1, j]
				# assert mid_box.sum() >0, [ for_box, mid_box, back_box ]
				# cur_patch = cur_img[i, :, int(mid_box[1]):int(mid_box[3])+1, int(mid_box[2]):int(mid_box[4])+1]
				# cur_patch = F.interpolate(cur_patch.unsqueeze(0), size=(self.H, self.W), mode='bilinear', align_corners=True)

				# forward check
				for_box = bboxes[i, 0, j]
				assert for_box.sum() > 0, [ for_box, back_box ] # for obj exist
				for_patch = for_img[i, :, int(for_box[1]):int(for_box[3])+1, int(for_box[2]):int(for_box[4])+1]
				for_patch = F.interpolate(for_patch.unsqueeze(0), size=(self.H, self.W), mode='bilinear', align_corners=True)
				# backward check
				back_box = bboxes[i, 2, j]
				assert back_box.sum() > 0, [ for_box,  back_box ] # for obj exist
				back_patch = back_img[i, :, int(back_box[1]):int(back_box[3])+1, int(back_box[2]):int(back_box[4])+1]
				back_patch = F.interpolate(back_patch.unsqueeze(0), size=(self.H, self.W), mode='bilinear', align_corners=True)
				track_patches.append(torch.cat([for_patch, back_patch], dim=1))
		track_patches = torch.cat(track_patches, dim=0)

		x_1 = self.encoder_1(track_patches)
		x_2 = self.encoder_2(x_1)
		x_3 = self.encoder_3(x_2)
		x_4 = self.encoder_4(x_3)

		out = self.bottom_layer(x_4)
		
		out = self.up_4(out)
		out = self.decoder_4(torch.cat([out, x_4], dim=1))		
		out = self.up_3(out)
		out = self.decoder_3(torch.cat([out, x_3], dim=1))		
		out = self.up_2(out)
		# print(out.size(), x_2.size())
		out = self.decoder_2(torch.cat([out, x_2], dim=1))		
		out = self.up_1(out)
		out = self.decoder_1(torch.cat([out, x_1], dim=1))

		out = out.view(bs, TRACK_NUM, 32, 64, 64)

		track_out_patches = torch.zeros(bs, TRACK_NUM, 32, self.args.input_h, self.args.input_w).cuda(input.get_device())
		track_out_mask = torch.zeros(bs, 1, self.args.input_h, self.args.input_w).cuda(input.get_device())
		for i in range(bs):
			for j in range(TRACK_NUM):
				mid_box = bboxes[i, 1, j]
				assert mid_box.sum() >0, mid_box
				location = [ int(mid_box[1]), int(mid_box[2]+1), int(mid_box[3]), int(mid_box[4]+1) ]
				track_out_mask[i, :, location[0]:location[2], location[1]:location[3]] = 1
				track_out_patches[i, j,:, location[0]:location[2], location[1]:location[3]] = \
					F.interpolate(out[i, j:j+1], size=(location[2]-location[0], location[3]-location[1]), mode='bilinear', align_corners=True).squeeze(0)
		track_out_patches = track_out_patches.view(bs, TRACK_NUM*32, self.args.input_h, self.args.input_w)
		track_out_patches = self.track_fusion_layer(track_out_patches)

		out = self.fusion_layer(torch.cat([track_out_patches, coarse_rgb, coarse_seg, track_out_mask], dim=1))

		rgb_out = self.rgb_out_layer(out)  
		seg_out = self.seg_out_layer(out)  

		return rgb_out, seg_out, None, torch.tensor([0]).cuda(self.args.rank)





class TrackGenV2(nn.Module):
	def __init__(self, args):
		super(TrackGenV2, self).__init__()
		self.args = args

		self.encoder_1 = nn.Sequential(
				nn.Conv2d(46+4, 32, 3, 1, 1),
				nn.LeakyReLU(0.2,inplace=True),
				nn.Conv2d(32, 32, 3, 1, 1),
				nn.LeakyReLU(0.2,inplace=True),
				nn.Conv2d(32, 32, 3, 1, 1),
				nn.LeakyReLU(0.2,inplace=True)			# 64x64
			)
		self.encoder_2 = nn.Sequential(
				nn.Conv2d(32, 64, 3, 2, 1),
				nn.LeakyReLU(0.2,inplace=True),
				nn.Conv2d(64, 64, 3, 1, 1),
				nn.LeakyReLU(0.2,inplace=True),
				nn.Conv2d(64, 64, 3, 1, 1),
				nn.LeakyReLU(0.2,inplace=True)			# 32x32
			)
		self.encoder_3 = nn.Sequential(
				nn.Conv2d(64, 128, 3, 2, 1),
				nn.LeakyReLU(0.2,inplace=True),
				nn.Conv2d(128, 128, 3, 1, 1),
				nn.LeakyReLU(0.2,inplace=True),
				nn.Conv2d(128, 128, 3, 1, 1),
				nn.LeakyReLU(0.2,inplace=True)			# 16x16
			)
		self.encoder_4 = nn.Sequential(
				nn.Conv2d(128, 128, 3, 2, 1),
				nn.LeakyReLU(0.2,inplace=True),
				nn.Conv2d(128, 128, 3, 1, 1),
				nn.LeakyReLU(0.2,inplace=True),
				nn.Conv2d(128, 128, 3, 1, 1),
				nn.LeakyReLU(0.2,inplace=True)			# 8x8
			)

		self.bottom_layer = nn.Sequential(
				nn.Conv2d(128, 256, 3, 2, 1),
				nn.LeakyReLU(0.2,inplace=True),
				nn.Conv2d(256, 256, 3, 1, 1),
				nn.LeakyReLU(0.2,inplace=True),
				nn.Conv2d(256, 256, 3, 1, 1),
				nn.LeakyReLU(0.2,inplace=True)			# 4x4
			)

		self.up_4 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True))
		self.decoder_4 = nn.Sequential(
				nn.Conv2d(128*2, 128, 3, 1, 1),
				nn.LeakyReLU(0.2,inplace=True),
				nn.Conv2d(128, 128, 3, 1, 1),
				nn.LeakyReLU(0.2,inplace=True),
				nn.Conv2d(128, 128, 3, 1, 1),
				nn.LeakyReLU(0.2,inplace=True)			# 8x8
			)
		self.up_3 = nn.Sequential(nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True))
		self.decoder_3 = nn.Sequential(
				nn.Conv2d(128*2, 128, 3, 1, 1),
				nn.LeakyReLU(0.2,inplace=True),
				nn.Conv2d(128, 128, 3, 1, 1),
				nn.LeakyReLU(0.2,inplace=True),
				nn.Conv2d(128, 128, 3, 1, 1),
				nn.LeakyReLU(0.2,inplace=True)			# 16x16
			)
		self.up_2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True))
		self.decoder_2 = nn.Sequential(
				nn.Conv2d(64*2, 64, 3, 1, 1),
				nn.LeakyReLU(0.2,inplace=True),
				nn.Conv2d(64, 64, 3, 1, 1),
				nn.LeakyReLU(0.2,inplace=True),
				nn.Conv2d(64, 64, 3, 1, 1),
				nn.LeakyReLU(0.2,inplace=True)			# 32x32
			)
		self.up_1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True))
		self.decoder_1 = nn.Sequential(
				nn.Conv2d(32*2, 32, 3, 1, 1),
				nn.LeakyReLU(0.2,inplace=True),
				nn.Conv2d(32, 32, 3, 1, 1),
				nn.LeakyReLU(0.2,inplace=True),
				nn.Conv2d(32, 32+2, 3, 1, 1)
				# nn.LeakyReLU(0.2,inplace=True)			# 64x64
			)

		TRACK_NUM = self.args.num_track_per_img

		self.track_fusion_layer = nn.Sequential(
				nn.Conv2d(32*TRACK_NUM, 128, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(128, 64, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(64, 32, 3, 1, 1)
			)

		self.fusion_layer = nn.Sequential(
				nn.Conv2d(32+3+20+1, 48, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(48, 48, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(48, 32, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(32, 32, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(32, 32, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True)
			)

		self.rgb_out_layer = nn.Sequential(
				nn.Conv2d(32, 32, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(32, 3, 3, 1, 1)
			)
		self.seg_out_layer = nn.Sequential(
				nn.Conv2d(32, 32, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(32, 20, 3, 1, 1)
			)

		self.H = 64
		self.W = 64

		self.track_num = self.args.num_track_per_img
		img_h = self.args.input_h
		img_w = self.args.input_w
		w_t = torch.matmul(
			torch.ones(img_h, 1), torch.linspace(-1.0, 1.0, img_w).view(1, img_w)).unsqueeze(0)
		h_t = torch.matmul(
			torch.linspace(-1.0, 1.0, img_h).view(img_h, 1), torch.ones(1, img_w)).unsqueeze(0)
		self.img_coord = torch.cat([h_t, w_t], dim=0).cuda(self.args.rank) #(2, h, w)

	def forward(self, input, coarse_rgb, coarse_seg, bboxes, gt_bbox_for_loss=False):
		bs = input.size(0)
		for_base_img_coord = self.img_coord.repeat(bs, 1, 1, 1)
		back_base_img_coord = self.img_coord.repeat(bs, 1, 1, 1)
		TRACK_NUM = self.track_num
		for_img  = torch.cat([input[:,:3], input[:,6:26]], dim=1)
		for_coord_img  = torch.cat([for_base_img_coord, for_img], dim=1)
		back_img = torch.cat([input[:,3:6], input[:,26:46]], dim=1)
		back_coord_img = torch.cat([back_base_img_coord, back_img], dim=1)

		track_patches = []
		for i in range(bs):
			for j in range(TRACK_NUM):
				# forward check
				for_box = bboxes[i, 0, j]
				assert for_box.sum() > 0, [ for_box, back_box ] # for obj exist
				for_patch = for_coord_img[i, :, int(for_box[1]):int(for_box[3])+1, int(for_box[2]):int(for_box[4])+1]
				for_patch = F.interpolate(for_patch.unsqueeze(0), size=(self.H, self.W), mode='bilinear', align_corners=True)
				# backward check
				back_box = bboxes[i, 2, j]
				assert back_box.sum() > 0, [ for_box,  back_box ] # for obj exist
				back_patch = back_coord_img[i, :, int(back_box[1]):int(back_box[3])+1, int(back_box[2]):int(back_box[4])+1]
				back_patch = F.interpolate(back_patch.unsqueeze(0), size=(self.H, self.W), mode='bilinear', align_corners=True)
				track_patches.append(torch.cat([for_patch, back_patch], dim=1))
		track_patches = torch.cat(track_patches, dim=0)

		x_1 = self.encoder_1(track_patches)
		x_2 = self.encoder_2(x_1)
		x_3 = self.encoder_3(x_2)
		x_4 = self.encoder_4(x_3)

		out = self.bottom_layer(x_4)
		
		out = self.up_4(out)
		out = self.decoder_4(torch.cat([out, x_4], dim=1))		
		out = self.up_3(out)
		out = self.decoder_3(torch.cat([out, x_3], dim=1))		
		out = self.up_2(out)

		out = self.decoder_2(torch.cat([out, x_2], dim=1))		
		out = self.up_1(out)
		out = self.decoder_1(torch.cat([out, x_1], dim=1))

		out = out.view(bs, TRACK_NUM, 32+2, 64, 64)
		### compute out bbox loc
		out_loc_ori = out[:,:,:2].view(bs*TRACK_NUM, 2, 64, 64)
		out_loc = out_loc_ori.clamp(-1,1)
		# calculate loss
		# if not gt_bbox_for_loss:
		# 	loc_diff loss does not work
		# 	out_loc_h_diff_loss = F.relu(out_loc[:, 0, 1:, :] - out_loc[:, 0, :-1, :]).mean(dim=(1,2))
		# 	out_loc_w_diff_loss = F.relu(out_loc[:, 1, :, 1:] - out_loc[:, 1, :, :-1]).mean(dim=(1,2))
		# 	out_loc_diff_loss = ((out_loc_h_diff_loss + out_loc_w_diff_loss)/2).mean()


		# out_loc = F.avg_pool2d(out_loc, kernel_size=9, stride=1, padding=9//2, count_include_pad=False)
		out_loc_center = out_loc.mean(dim=(2,3), keepdim=False) # (y, x)

		out_loc_hmax = out_loc[:,0,-1,:].mean(dim=1,keepdim=True)
		out_loc_hmin = out_loc[:,0,0,:].mean(dim=1,keepdim=True)
		out_loc_hrange = (out_loc_hmax - out_loc_hmin)
		if not gt_bbox_for_loss:
			loc_range_loss = 2-out_loc_hrange
		out_loc_hrange = F.relu(out_loc_hrange)
		# if self.args.rank==0:
		# 	print('hmax', out_loc_hmax[0])
		# 	print('hmin', out_loc_hmin[0])

		out_loc_wmax = out_loc[:,1,:,-1].mean(dim=1,keepdim=True)
		out_loc_wmin = out_loc[:,1,:,0].mean(dim=1,keepdim=True)
		out_loc_wrange = (out_loc_wmax - out_loc_wmin)
		if not gt_bbox_for_loss:
			loc_range_loss += (2-out_loc_wrange)
			loc_range_loss = loc_range_loss.mean()
		out_loc_wrange = F.relu(out_loc_wrange)
		# print('wrange', out_loc_wrange[0])



		out_loc_bbox_h1 = (((out_loc_center[:,0:1]-out_loc_hrange/2) + 1)/2 *self.args.input_h).clamp(0, self.args.input_h-1)
		out_loc_bbox_h2 = (((out_loc_center[:,0:1]+out_loc_hrange/2) + 1)/2 *self.args.input_h).clamp(0, self.args.input_h-1)
		out_loc_bbox_w1 = (((out_loc_center[:,1:]-out_loc_wrange/2) + 1)/2 *self.args.input_w).clamp(0, self.args.input_w-1)
		out_loc_bbox_w2 = (((out_loc_center[:,1:]+out_loc_wrange/2) + 1)/2 *self.args.input_w).clamp(0, self.args.input_w-1)
		out_loc_bbox_h1w1h2w2 = torch.cat([out_loc_bbox_h1, out_loc_bbox_w1, out_loc_bbox_h2, out_loc_bbox_w2], 
														dim=1).view(bs, TRACK_NUM, 4)#(bs*track, 4)

		out = out[:,:,2:] 

		if gt_bbox_for_loss:
			gt_location = []
		mid_base_img_coord = self.img_coord.repeat(bs, 1, 1, 1)
		track_out_patches = torch.zeros(bs, TRACK_NUM, 32, self.args.input_h, self.args.input_w).cuda(self.args.rank)
		track_out_mask = torch.zeros(bs, 1, self.args.input_h, self.args.input_w).cuda(self.args.rank)
		for i in range(bs):
			for j in range(TRACK_NUM):
				if gt_bbox_for_loss:
					gt_box = bboxes[i, 1, j]
					gt_coord_patch = mid_base_img_coord[i, :, int(gt_box[1]):int(gt_box[3])+1, int(gt_box[2]):int(gt_box[4])+1]
					gt_coord_patch = F.interpolate(gt_coord_patch.unsqueeze(0), size=(self.H, self.W), mode='bilinear', align_corners=True)
					gt_location.append(gt_coord_patch)
				mid_box = out_loc_bbox_h1w1h2w2[i, j]
				assert mid_box.sum() >=0, mid_box
				location = [ int(mid_box[0]), int(mid_box[1]), int(mid_box[2]+1), int(mid_box[3]+1) ]
				track_out_mask[i, :, location[0]:location[2], location[1]:location[3]] = 1
				track_out_patches[i, j,:, location[0]:location[2], location[1]:location[3]] = \
					F.interpolate(out[i, j:j+1], size=(location[2]-location[0], location[3]-location[1]), mode='bilinear', align_corners=True).squeeze(0)
		if gt_bbox_for_loss:
			gt_location = torch.cat(gt_location, dim=0)
			loc_diff_loss = torch.abs(gt_location-out_loc_ori).mean()
		track_out_patches = track_out_patches.view(bs, TRACK_NUM*32, self.args.input_h, self.args.input_w)
		track_out_patches = self.track_fusion_layer(track_out_patches)

		out = self.fusion_layer(torch.cat([track_out_patches, coarse_rgb, coarse_seg, track_out_mask], dim=1))

		rgb_out = self.rgb_out_layer(out)  
		seg_out = self.seg_out_layer(out)
		# if self.args.rank==0:
		# 	print('bbox', out_loc_bbox_h1w1h2w2[0,0])
		if gt_bbox_for_loss:
			return rgb_out, seg_out, out_loc_bbox_h1w1h2w2, loc_diff_loss
		else:
			return rgb_out, seg_out, out_loc_bbox_h1w1h2w2, loc_range_loss






		
