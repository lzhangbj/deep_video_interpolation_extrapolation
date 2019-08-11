import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import nets


class InterGANNet(nn.Module):
	def __init__(self, args):
		super(InterGANNet, self).__init__()
		self.args = args
		self.coarse_model = nets.__dict__[args.coarse_model](args)
		if self.args.frame_disc:
			self.frame_disc_model = nets.__dict__[args.frame_disc_model](args)
		if self.args.frame_det_disc:
			self.frame_det_disc_model = nets.__dict__[args.frame_det_disc_model](args)
		if self.args.video_disc:
			self.video_disc_model = nets.__dict__[args.video_disc_model](args)
		if self.args.video_det_disc:
			self.video_det_disc_model = nets.__dict__[args.video_det_disc_model](args)
		if self.args.track_gen:
			self.track_gen_model = nets.__dict__[args.track_gen_model](args)

	def set_net_grad(self, net, flag=True):
		for p in net.parameters():
			p.requires_grad = flag

	def forward(self, input, seg=None, gt_x=None, gt_seg=None, bboxes=None):
		low_input = torch.cat([input, seg], dim=1) 
		coarse_rgb, coarse_seg, mu, var = self.coarse_model(low_input, gt_x, gt_seg)
		if self.args.track_gen:
			if 'V2' in self.args.track_gen_model:
				coarse_rgb, coarse_seg, gen_bbox, loc_diff_loss = self.track_gen_model(low_input, coarse_rgb, coarse_seg, bboxes, True)
			else:
				coarse_rgb, coarse_seg, gen_bbox, loc_diff_loss = self.track_gen_model(low_input, coarse_rgb, coarse_seg, bboxes)
		else:
			gen_bbox = None
			loc_diff_loss = torch.tensor([0]).cuda(self.args.rank)

		coarse_seg_softed = F.softmax(coarse_seg, dim=1)

		#### traing discriminator ####
		### frame discriminator ###
		if self.training:
			if self.args.frame_disc:
				D_fake_frame_prob = self.frame_disc_model(coarse_rgb.detach(), coarse_seg_softed.detach(), bboxes=bboxes)  
				D_real_frame_prob = self.frame_disc_model(gt_x, gt_seg, bboxes=bboxes)
			else:
				D_fake_frame_prob = None
				D_real_frame_prob = None
			if self.args.frame_det_disc:
				### frame det discriminator ###
				D_fake_frame_det_prob = self.frame_det_disc_model(coarse_rgb.detach(), coarse_seg_softed.detach(), bboxes=bboxes)  
				D_real_frame_det_prob = self.frame_det_disc_model(gt_x, gt_seg, bboxes=bboxes)
			else:
				D_fake_frame_det_prob = None
				D_real_frame_det_prob = None
			### video discriminator ###
			if self.args.video_disc:
				D_fake_video_prob = self.video_disc_model(coarse_rgb.detach(), coarse_seg_softed.detach(), input, seg, bboxes=bboxes)  
				D_real_video_prob = self.video_disc_model(gt_x, gt_seg, input, seg, bboxes=bboxes)
			else:
				D_fake_video_prob=None
				D_real_video_prob=None

			if self.args.video_det_disc:
				### video discriminator ###
				D_fake_video_det_prob = self.video_det_disc_model(coarse_rgb.detach(), coarse_seg_softed.detach(), input, seg, bboxes=bboxes, gt_x=gt_x)  
				D_sync_fake_video_det_prob = self.video_det_disc_model(gt_x, gt_seg, input, seg, bboxes=bboxes, sync_neg=True, gt_x=gt_x)  
				D_real_video_det_prob = self.video_det_disc_model(gt_x, gt_seg, input, seg, bboxes=bboxes, gt_x=gt_x)
			else:
				D_fake_video_det_prob = None
				D_real_video_det_prob = None
				D_sync_fake_video_det_prob = None
			#### training generator ####  
			### frame ###
			if self.args.frame_disc:
				self.set_net_grad(self.frame_disc_model, False)   
				G_fake_frame_prob = self.frame_disc_model(coarse_rgb, coarse_seg_softed, bboxes=bboxes)
				if self.training:
					self.set_net_grad(self.frame_disc_model, True) 
			else:
				G_fake_frame_prob=None
			if self.args.frame_det_disc: 
				### frame det ###
				self.set_net_grad(self.frame_det_disc_model, False)   
				G_fake_frame_det_prob = self.frame_det_disc_model(coarse_rgb, coarse_seg_softed, bboxes=bboxes)
				if self.training:
					self.set_net_grad(self.frame_det_disc_model, True)  
			else:
				G_fake_frame_det_prob = None
			### video ###
			if self.args.video_disc:
				self.set_net_grad(self.video_disc_model, False)   
				G_fake_video_prob = self.video_disc_model(coarse_rgb, coarse_seg_softed, input, seg, bboxes=bboxes)
				if self.training:
					self.set_net_grad(self.video_disc_model, True)  
			else:
				G_fake_video_prob=None
			if self.args.video_det_disc:
				### video det ###
				self.set_net_grad(self.video_det_disc_model, False)   
				G_fake_video_det_prob = self.video_det_disc_model(coarse_rgb, coarse_seg_softed, input, seg, bboxes=bboxes, gt_x=gt_x)
				if self.training:
					self.set_net_grad(self.video_det_disc_model, True)
			else:
				G_fake_video_det_prob = None


			return coarse_rgb, coarse_seg, \
					mu, var, \
					D_fake_frame_prob, D_real_frame_prob, \
					D_fake_video_prob, D_real_video_prob, \
					G_fake_frame_prob, G_fake_video_prob, \
					D_fake_frame_det_prob, D_real_frame_det_prob, \
					D_fake_video_det_prob, D_sync_fake_video_det_prob, D_real_video_det_prob, \
					G_fake_frame_det_prob, G_fake_video_det_prob, gen_bbox, loc_diff_loss

		else:
			return coarse_rgb, coarse_seg, \
					mu, var, \
					0, 0, \
					0, 0, \
					0, 0, \
					0, 0, \
					0, 0, \
					0, 0, gen_bbox, loc_diff_loss
