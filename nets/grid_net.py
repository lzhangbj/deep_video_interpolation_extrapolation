import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from nets.vgg import *
from net_utils import *
from losses import *

MODE_LIST = ['s2s', 'x2x', 'xs2s', 'xs2x']

mean = torch.FloatTensor([0.485, 0.456, 0.406]).view([1,3,1,1])
std = torch.FloatTensor([0.229, 0.224, 0.225]).view([1,3,1,1])


class Lateral(nn.Module):
	def __init__(self, in_channel, kernel_size, out_channel=None, shortcut_conv=False, prelu=True):
		super(Lateral, self).__init__()
		if out_channel is None:
			out_channel = in_channel
		self.in_channel = in_channel
		self.out_channel = out_channel
		self.kernel_size = kernel_size
		if prelu:
			self.net = nn.Sequential(
				nn.PReLU(),
				nn.Conv2d(in_channel, out_channel, kernel_size, stride=1, padding=kernel_size//2),
				nn.PReLU(),
				nn.Conv2d(out_channel, out_channel, kernel_size, stride=1, padding=kernel_size//2)
				)		
		else:
			self.net = nn.Sequential(
				nn.Conv2d(in_channel, out_channel, kernel_size, stride=1, padding=kernel_size//2),
				nn.PReLU(),
				nn.Conv2d(out_channel, out_channel, kernel_size, stride=1, padding=kernel_size//2),
				nn.PReLU(),
				nn.Conv2d(out_channel, out_channel, kernel_size, stride=1, padding=kernel_size//2)
				)	
		if (self.out_channel != self.in_channel) and shortcut_conv:
			self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride=1, padding=kernel_size//2)
		# normal_init_net_conv(self.net)
		self.shortcut_conv = shortcut_conv
			
	def forward(self, input):
		assert input.size(1) == self.in_channel, [ input.size(1), self.in_channel ]
		if self.shortcut_conv:
			if self.out_channel != self.in_channel:
				return self.net(input) + self.conv(input)
			else:
				return self.net(input) + input
		else:
			return self.net(input) 

class Upsample(nn.Module):
	def __init__(self, in_channel, out_channel, kernel_size=3):
		super(Upsample, self).__init__()
		self.in_channel = in_channel
		self.out_channel = out_channel
		self.kernel_size = kernel_size
		self.net = nn.Sequential(
			nn.PReLU(),
			nn.Conv2d(in_channel, out_channel, kernel_size, stride=1, padding=kernel_size//2),
			nn.PReLU(),
			nn.Conv2d(out_channel, out_channel, kernel_size, stride=1, padding=kernel_size//2)
			)
		# normal_init_net_conv(self.net)

	def forward(self, input):
		assert input.size(1) == self.in_channel, [ input.size(1), self.in_channel ]
		return self.net(F.interpolate(input, scale_factor=2, mode='bilinear', align_corners=True))


class Downsample(nn.Module):
	def __init__(self, in_channel, out_channel, kernel_size=3):
		super(Downsample, self).__init__()
		self.in_channel = in_channel
		self.out_channel = out_channel
		self.kernel_size = kernel_size
		self.net = nn.Sequential(
			nn.PReLU(),
			nn.Conv2d(in_channel, out_channel, kernel_size, stride=2, padding=kernel_size//2),
			nn.PReLU(),
			nn.Conv2d(out_channel, out_channel, kernel_size, stride=1, padding=kernel_size//2)
			)
		# normal_init_net_conv(self.net)

	def forward(self, input):
		assert input.size(1) == self.in_channel, [ input.size(1), self.in_channel ]
		return self.net(input)


class Downflow(nn.Module):
	def __init__(self, in_channels, kernel_size=3):
		super(Downflow, self).__init__()
		self.in_channels = in_channels
		self.row0 = Lateral(in_channels[0], kernel_size, shortcut_conv=False)
		self.row1 = Lateral(in_channels[1], kernel_size, shortcut_conv=False)
		self.row2 = Lateral(in_channels[2], kernel_size, shortcut_conv=False)
		self.down01 = Downsample(in_channels[0], in_channels[1])
		self.down12 = Downsample(in_channels[1], in_channels[2])

	def forward(self, row0_input, row1_input, row2_input):
		assert row0_input.size(1) == self.in_channels[0], [ row0_input.size(1), self.in_channels[0] ]
		assert row1_input.size(1) == self.in_channels[1], [ row1_input.size(1), self.in_channels[1] ]
		assert row2_input.size(1) == self.in_channels[2], [ row2_input.size(1), self.in_channels[2] ]

		row0_output = self.row0(row0_input)
		row1_output = self.row1(row1_input)
		row2_output = self.row2(row2_input)

		row1_output = self.down01(row0_output) + row1_output
		row2_output = self.down12(row1_output) + row2_output

		return row0_output, row1_output, row2_output


class Upflow(nn.Module):
	def __init__(self, in_channels, kernel_size=3):
		super(Upflow, self).__init__()
		self.in_channels = in_channels
		self.row0 = Lateral(in_channels[0], kernel_size, shortcut_conv=False)
		self.row1 = Lateral(in_channels[1], kernel_size, shortcut_conv=False)
		self.row2 = Lateral(in_channels[2], kernel_size, shortcut_conv=False)
		self.up10 = Upsample(in_channels[1], in_channels[0])
		self.up21 = Upsample(in_channels[2], in_channels[1])

	def forward(self, row0_input, row1_input, row2_input):
		assert row0_input.size(1) == self.in_channels[0], [ row0_input.size(1), self.in_channels[0] ]
		assert row1_input.size(1) == self.in_channels[1], [ row1_input.size(1), self.in_channels[1] ]
		assert row2_input.size(1) == self.in_channels[2], [ row2_input.size(1), self.in_channels[2] ]

		row0_output = self.row0(row0_input)
		row1_output = self.row1(row1_input)
		row2_output = self.row2(row2_input)

		row1_output = self.up21(row2_output) + row1_output
		row0_output = self.up10(row1_output) + row0_output
		
		return row0_output, row1_output, row2_output


class GridNet(nn.Module):
	def __init__(self, n_channels, n_classes, mode='s2s', split_tail=False, seg_id=False):
		super(GridNet, self).__init__()
		self.mode = mode
		self.CELoss = nn.CrossEntropyLoss()
		self.SSIMLoss = SSIM()
		self.seg_act = nn.Softmax(dim=1)
		self.split_tail = split_tail
		self.seg_id = seg_id
		if mode == 'x2x':
			self.in_channel = 3*2
			self.out_channel = 3
		elif mode == 'xs2x':
			if not seg_id:
				self.in_channel = (3+n_classes)*2
			else:
				self.in_channel = (3+1)*2
			self.out_channel = 3
		elif mode == 's2s':
			if not seg_id:
				self.in_channel = n_classes*2
				self.out_channel = n_classes
			else:
				self.in_channel = 2
				self.out_channel = n_classes				
		elif mode == 'xs2s':
			if not seg_id:
				self.in_channel = (3+n_classes)*2
				self.out_channel = n_classes
			else:
				self.in_channel = (3+1)*2
				self.out_channel = n_classes
		elif mode == 'xs2xs':
			if not split_tail:
				if not seg_id:
					self.in_channel = (3+n_classes)*2
					self.out_channel = (3+n_classes)
				else:
					self.in_channel = (3+1)*2
					self.out_channel = 3+n_classes
			else:
				if not seg_id:
					self.in_channel = (3+n_classes)*2
					self.out_channel = 3
					self.out_channel_seg = n_classes
				else:
					self.in_channel = (3+1)*2
					self.out_channel = 3
					self.out_channel_seg = n_classes
		elif mode == 'wing':
			if not seg_id:
				self.in_channel = (3+n_classes)*2 + 3
				self.out_channel = n_classes
			else:
				self.in_channel = (3 + 1)*2 + 3
				self.out_channel = n_classes

		else:
			raise Exception("mode doesnt exist !")

		self.n_channels = n_channels
		self.head = Lateral(self.in_channel, 3, n_channels[0], shortcut_conv=True, prelu=False)

		# nn.Sequential(
		# 	nn.PReLU(),
		# 	nn.Conv2d(self.in_channel, n_channels[0], 3, stride=1, padding=1),
		# 	nn.PReLU(),
		# 	nn.Conv2d(n_channels[0], n_channels[0], 3, stride=1, padding=1)
		# 	)

		self.neck_down01 = Downsample(n_channels[0], n_channels[1], 3)
		self.neck_down12 = Downsample(n_channels[1], n_channels[2], 3)

		self.body_down0 = Downflow(n_channels, 3)
		self.body_down1 = Downflow(n_channels, 3)

		self.body_up0 = Upflow(n_channels, 3)
		self.body_up1 = Upflow(n_channels, 3)
		self.body_up2 = Upflow(n_channels, 3)

		self.tail = Lateral(n_channels[0], 3, self.out_channel, shortcut_conv=False, prelu=True)
		if self.split_tail:
			self.tail_seg = Lateral(n_channels[0], 3, self.out_channel_seg, shortcut_conv=False, prelu=True)

		if self.mode[-1] == 'x' or self.mode == 'xs2xs':
			vgg19 = torchvision.models.vgg19(pretrained=True)
			self.vgg_net = my_vgg(vgg19)
			for param in self.vgg_net.parameters():
				param.requires_grad = False


	def GDLLoss(self, input, gt):
		bs, c, h, w = input.size()

		w_gdl = torch.abs(input[:,:,:,1:] - input[:,:,:,:w-1])
		h_gdl = torch.abs(input[:,:,1:,:] - input[:,:,:h-1,:])

		gt_w_gdl = torch.abs(gt[:,:,:,1:] - gt[:,:,:,:w-1])
		gt_h_gdl = torch.abs(gt[:,:,1:,:] - gt[:,:,:h-1,:])
		
		loss = torch.mean(torch.abs(w_gdl-gt_w_gdl)) + torch.mean(torch.abs(h_gdl-gt_h_gdl))
		return loss

	def _normalize(self, x):
		gpu_id = x.get_device()
		return (x - mean.cuda(gpu_id)) / std.cuda(gpu_id)

	def VGGLoss(self, pred_feat, true_feat):
		loss = 0
		for i in range(len(pred_feat)):
			loss += (true_feat[i] - pred_feat[i]).abs().mean()
		return loss/len(pred_feat)

	def L1Loss(self, input, gt):
		theta = 0.001
		# fg_indices = [4,5,6,7,11,12,13,14,15,16,17,18]
		diff = (input-gt)**2
		# diff[:,fg_indices] = 4*diff[:,fg_indices]
		return torch.sqrt(diff + theta**2)


	def forward(self, input, gt=None):
		assert input.size(1) == self.in_channel, [input.size(), self.in_channel]

		# change to anqi test method
		# if self.mode=='xs2xs':
		# 	input[:,:6] = postprocess_output(input[:,:6])
		# 	gt[:,:3] = postprocess_output(gt[:,:3])

		row0_out = self.head(input)
		row1_out = self.neck_down01(row0_out)
		row2_out = self.neck_down12(row1_out)

		row0_out, row1_out, row2_out = self.body_down0(row0_out, row1_out, row2_out)
		row0_out, row1_out, row2_out = self.body_down1(row0_out, row1_out, row2_out)
		row0_out, row1_out, row2_out = self.body_up0(row0_out, row1_out, row2_out)
		row0_out, row1_out, row2_out = self.body_up1(row0_out, row1_out, row2_out)
		out, row1_out, row2_out = self.body_up2(row0_out, row1_out, row2_out)

		if not self.split_tail:
			if self.mode =='wing':
				out = self.seg_act(self.tail(out))
			elif self.mode[-1] != 's':
				out = F.tanh(self.tail(out))
				# print("hhhh")
			else:
				out = self.seg_act(self.tail(out))
				# print(torch.nonzero(out).size(0))
		else:
			assert self.mode=='xs2xs'
			out_seg =self.tail_seg(out)
			out = F.tanh(self.tail(out))

		l1_loss = None
		gdl_loss = None
		vgg_loss = None
		ce_loss = None
		ssim_loss = None
			

		if self.training:
			# if self.mode.split('2')[1] in ['x','xs'] or (not self.seg_id ):
			if self.mode[-1] == 'x':
				gdl_loss = self.GDLLoss(preprocess_norm(out), preprocess_norm(gt))
				l1_loss = self.L1Loss(preprocess_norm(out), preprocess_norm(gt))
				ssim_loss =  self.SSIMLoss(preprocess_norm(out), preprocess_norm(gt)).mean()

				predict_feat = self.vgg_net(preprocess_norm(out))
				true_feat = self.vgg_net(preprocess_norm(gt))
				vgg_loss = self.VGGLoss(predict_feat, true_feat)

			elif self.mode == 'wing' or self.mode.split('2')[1] == 's' :
				# gdl_loss = self.GDLLoss(out, gt)
				# l1_loss = self.L1Loss(out, gt)
				if not self.seg_id:
					ce_loss = self.CELoss(out, torch.argmax(gt, dim=1))
				else:
					ce_loss = self.CELoss(out, gt.squeeze(1).long())
					# gdl_loss = self.GDLLoss(out, gt)
					# l1_loss = self.L1Loss(out, gt)
				vgg_loss = None
			elif self.mode == 'xs2xs': ####################### try here
				# if self.ce:
				# if self.seg_id:
				# gdl_loss = self.GDLLoss(out, gt[:,:3])
				# l1_loss = self.L1Loss(out, gt[:,:3])
				# ssim_loss = 1 - self.SSIMLoss(postprocess_output(out), postprocess_output(gt[:, :3])).mean()
				gdl_loss = self.GDLLoss(preprocess_norm(out), preprocess_norm(gt[:,:3]))
				l1_loss = self.L1Loss(preprocess_norm(out), preprocess_norm(gt[:,:3]))
				ssim_loss =  self.SSIMLoss(preprocess_norm(out), preprocess_norm(gt[:, :3])).mean()
				if not self.seg_id:
					ce_loss =  self.CELoss(out_seg, torch.argmax(gt[:, 3:], dim=1))
				else:
					ce_loss =  self.CELoss(out_seg, gt[:, 3:].squeeze(1).long())
				# else:
				# 	gdl_loss = self.GDLLoss(out, gt)
				# 	l1_loss = self.L1Loss(out, gt)
				# else:
				# gdl_loss = self.GDLLoss(torch.cat([out, out_seg], dim=1), gt)
				# l1_loss = self.L1Loss(torch.cat([out, out_seg], dim=1), gt)
				predict_feat = self.vgg_net(preprocess_norm(out))
				true_feat = self.vgg_net(preprocess_norm(gt[:, :3]))
				
				vgg_loss = self.VGGLoss(predict_feat, true_feat)
			# 	else:
			# 		vgg_loss = None
			# else:
			# 	vgg_loss = None

			# if self.seg_id:
			# 	if  self.mode.split('2')[1] == 's':
			# 		ce_loss = self.celoss(out, gt)
			# 	else:  # xs2xs
			# 		ce_loss = self.celoss(out_seg, gt[:, 3].long())
			# else:
			# 	ce_loss = None

		if self.mode == 'xs2xs' and self.split_tail:
			out_seg =  self.seg_act(out_seg)
			out = torch.cat([out, out_seg], dim=1)

			### todo laplacian pyramid loss for image ###

		return out, l1_loss, gdl_loss, vgg_loss, ce_loss, ssim_loss



