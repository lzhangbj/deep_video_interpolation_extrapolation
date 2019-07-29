import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import nets


class InterRefineNet(nn.Module):
	def __init__(self, args):
		super(InterRefineNet, self).__init__()
		self.args = args
		self.coarse_model = nets.__dict__[args.coarse_model](args)
		self.refine_model = nets.__dict__[args.refine_model](args)

	def forward(self, input, seg=None, gt_seg=None):
		low_input = torch.cat([input, seg], dim=1) 
		coarse_rgb, coarse_seg = self.coarse_model(low_input)
		coarse_seg_softed = F.softmax(coarse_seg, dim=1).detach()
		if self.args.split == 'val' and self.args.with_gt_seg:
			coarse_seg_softed=gt_seg
		seg_encoded = [
				self.coarse_model.seg_encoder(seg[:, :20]).detach(),
				self.coarse_model.seg_encoder(seg[:, 20:40].detach())
			]
		encoded_feat = torch.cat([input] + seg_encoded, dim=1)
		refine_rgbs = self.refine_model(coarse_rgb.detach().clamp(-1,1), coarse_seg_softed, encoded_feat)
		refine_rgbs = [img.clamp_(-10,10) for img in refine_rgbs]
		return coarse_rgb, coarse_seg, refine_rgbs

class InterStage3Net(nn.Module):
	def __init__(self, args):
		super(InterStage3Net, self).__init__()
		self.args = args
		self.coarse_model = nets.__dict__[args.coarse_model](args)
		self.refine_model = nets.__dict__[args.refine_model](args)
		self.stage3_model = nets.__dict__[args.stage3_model](args)

	def forward(self, input, seg=None, gt_seg=None):
		low_input = torch.cat([input, seg], dim=1) 
		coarse_rgb, coarse_seg = self.coarse_model(low_input)
		coarse_seg_softed = F.softmax(coarse_seg, dim=1).detach()
		if self.args.split == 'val' and self.args.with_gt_seg:
			coarse_seg_softed=gt_seg
		seg_encoded = [
				self.coarse_model.seg_encoder(seg[:, :20]).detach(),
				self.coarse_model.seg_encoder(seg[:, 20:40].detach())
			]
		encoded_feat = torch.cat([input] + seg_encoded, dim=1)
		refine_rgbs = self.refine_model(coarse_rgb.detach().clamp(-1,1), coarse_seg_softed, encoded_feat)
		refine_rgbs = [img.clamp_(-1,1) for img in refine_rgbs]
		re_refine_rgbs, flow_maps = self.stage3_model(refine_rgbs[-1].detach(), coarse_seg_softed, input, seg)
		re_refine_rgbs = [img.clamp_(-10,10) for img in re_refine_rgbs]
		return coarse_rgb, coarse_seg, refine_rgbs, re_refine_rgbs, flow_maps