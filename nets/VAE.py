import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from torch.autograd import Variable as Vb
from nets.vgg import RefineNet, my_vgg
from utils.net_utils import *
from losses import *
from nets.SubNets import *
MODE_LIST = ['s2s', 'x2x', 'xs2s', 'xs2x']

mean = Vb(torch.FloatTensor([0.485, 0.456, 0.406]).view([1,3,1,1]))
std = Vb(torch.FloatTensor([0.229, 0.224, 0.225]).view([1,3,1,1]))

class VAE(nn.Module):
	def __init__(self, args):
		super(VAE, self).__init__()

		self.args = args
		self.seg_encoder = SegEncoder(in_dim=20, out_dim=4)

		# BG
		self.flow_encoder_bg = FlowEncoder(args,in_dim=3*(args.vid_length+1)+args.seg_dim,latent_dim=128)
		# FG
		self.flow_encoder_fg = FlowEncoder(args,in_dim=3*(args.vid_length+1)+args.seg_dim,latent_dim=896)

		self.encoder = encoder(args)
		self.flow_decoder = decoder(args)


		self.zconv = convbase(256+48, 16*self.args.vid_length, 3, 1, 1)
		self.floww = FlowWrapper()
		self.fc = nn.Linear(1024, 1024)
		self.flownext = getflow()
		self.flowprev = getflow()
		self.get_mask = get_occlusion_mask()
		self.refine = True
		if self.refine:
			self.refine_net = RefineNet(num_channels=3)

		vgg19 = torchvision.models.vgg19(pretrained=True)
		self.vgg_net = my_vgg(vgg19)
		for param in self.vgg_net.parameters():
			param.requires_grad = False

	def reparameterize(self, mu, logvar):
		if self.training:
			std = logvar.mul(0.5).exp_()
			eps = Vb(std.data.new(std.size()).normal_())
			return eps.mul(std).add_(mu)
		else:
			return Vb(mu.data.new(mu.size()).normal_())

	def _normalize(self, x):
		gpu_id = x.get_device()
		return (x - mean.cuda(gpu_id)) / std.cuda(gpu_id)

	def forward(self, rgb_data, seg_data, bg_mask, fg_mask, noise_bg, z_m=None):
		args = self.args
		frame1 = rgb_data[:, 0]
		frame2 = rgb_data[:, 1:]

		# seg_encoded = torch.stack([self.seg_encoder(seg_data[:, i]) for i in range(args.vid_length+1)], dim=1)
		# fg_seg_encoded = seg_encoded*fg_mask
		# bg_seg_encoded = seg_encoded*bg_mask  
		seg_encoded = self.seg_encoder(seg_data[:,0])
		fg_seg_encoded = seg_encoded*fg_mask[:,0]
		bg_seg_encoded = seg_encoded*bg_mask[:,0]

		# seg1 = seg_encoded[:, 0]
		# seg2 = seg_encoded[:, 1:]

		# mask = torch.cat([bg_mask, fg_mask], 1)
		input = torch.cat([frame1, seg_encoded], 1)

		# Encoder Network --> encode input frames
		# bs*4, 32, 64, 64
		# 64, 32, 32
		# 128, 16, 16
		# 256, 8, 8
		enco1, enco2, enco3, codex = self.encoder(input)
		

		if z_m is None:

			y = torch.cat(
				[frame1, frame2.contiguous().view(-1, args.vid_length * 3, args.input_size[0], args.input_size[1])], 1)

			# Motion Network --> compute latent vector

			# BG
			mu_bg, logvar_bg = self.flow_encoder_bg(
				torch.cat(
						[y, bg_seg_encoded], 1)
				.contiguous())
			# FG
			mu_fg, logvar_fg = self.flow_encoder_fg(
				torch.cat(
						[y, fg_seg_encoded], 1)
						# [y, fg_seg_encoded.view(-1, (args.vid_length+1)*args.seg_dim, args.input_size[0], args.input_size[1])], 1)
				.contiguous())

			mu = torch.cat([mu_bg, mu_fg], 1)
			logvar = torch.cat([logvar_bg, logvar_fg], 1)

			z_m = self.reparameterize(mu, logvar)


		# (bs, 64, 8, 8) + (bs, 256, 8, 8)
		# codex = codex.repeat(4, 1)

		# z = z_m*codex + codex

		# code = self.zconv(torch.cat([self.fc(z).view(-1, 64, int(args.input_size[0]/16), int(args.input_size[1]/16)), codex], 1))

		codey = self.zconv(torch.cat([self.fc(z_m).view(-1, 48, int(args.input_size[0]/16), int(args.input_size[1]/16)), codex], 1)) # 64, 8, 8
		codex = torch.unsqueeze(codex, 2).repeat(1, 1, args.vid_length, 1, 1)  # bs,256,3,8,8
		codey = torch.cat(torch.chunk(codey.unsqueeze(2), args.vid_length, 1), 2)  # bs,16,3,8,8
		z = torch.cat(torch.unbind(torch.cat([codex, codey], 1), 2), 0)  # (256L, 272L, 8L, 8L)   272-256=16 
		# (bs*3, 256, 8, 8)

		# Flow Decoder Network --> decode latent vectors into flow fields.
		flow_deco4 = self.flow_decoder(enco1, enco2, enco3, z)  # (256, 64, 64, 64)
		flow = torch.cat(self.flownext(flow_deco4).unsqueeze(2).chunk(args.vid_length, 0), 2)  # (64, 2, 4, 128, 128)
		flowback = torch.cat(self.flowprev(flow_deco4).unsqueeze(2).chunk(args.vid_length, 0), 2) # (64, 2, 4, 128, 128)

		'''Compute Occlusion Mask'''
		masks = torch.cat(self.get_mask(flow_deco4).unsqueeze(2).chunk(args.vid_length, 0), 2)  # (64, 2, 4, 128, 128)
		mask_fw = masks[:, 0, ...]
		mask_bw = masks[:, 1, ...]

		'''Use mask before warpping'''
		output = warp(frame1, flow, args, self.floww, mask_fw)

		y_pred = output

		'''Go through the refine network.'''
		if self.refine:
			y_pred = refine(output, flow, mask_fw, self.refine_net, args, noise_bg)

		if self.training:
			prediction_vgg_feature = self.vgg_net(
				self._normalize(output.contiguous().view(-1, 3, args.input_size[0], args.input_size[1])))
			gt_vgg_feature = self.vgg_net(
				self._normalize(frame2.contiguous().view(-1, 3, args.input_size[0], args.input_size[1])))

			return output, y_pred, mu, logvar, flow, flowback, mask_fw, mask_bw, prediction_vgg_feature, gt_vgg_feature
		else:
			return output, y_pred, flow, flowback, mask_fw, mask_bw
