import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models


class B2SConv(nn.Module):
	def __init__(self, in_dim, med_dim, out_dim, n_conv=1):
		super(B2SConv, self).__init__()
		self.n_conv = n_conv
		for i in range(n_conv):
			indim = in_dim if i==0 else in_dim+med_dim
			outdim = outdim if i==n_conv-1 else med_dim
			setattr(self, 'conv'+str(i),
					nn.Sequential(
							# nn.AvgPool2d(scale),
							nn.Conv2d(indim, outdim, 3, 1, 1),
							nn.LeakyReLU(0.2, inplace=True)
							# nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
						)
				)

	def forward(self, ori_input):
		for i in range(self.n_conv):
			# possibly downsample ori input
			if i!=self.n_conv-1:
				scale = 2**(self.n_conv-1-i)
				scaled_ori_input = F.interpolate(ori_input,scale_factor=1/scale, mode='bilinear', align_corners=True)
			else:
				scaled_ori_input = ori_input
			# conv layer
			if i==0:
				output = getattr(self, 'conv'+str(i))(scaled_ori_input)
			else:
				output = getattr(self, 'conv'+str(i))( torch.cat([scaled_ori_input,output], dim=1) )
			# possibly upsample output 
			if i!=self.n_conv-1:
				output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=True)
		return output


class B2SConvModule(nn.Module):
	def __init__(self, in_dim, med_dim, out_dim, n_scales, n_conv=1):
		super(B2SConvModule, self).__init__()
		self.n_scales = n_scales
		for ind in range(n_scales):
			indim = in_dim if ind == 0 else in_dim+med_dim
			outdim = out_dim if ind == n_scales-1 else med_dim
			setattr(self, 'conv'+str(ind),
						B2SConv(indim, med_dim, outdim, n_conv=n_conv)
					)
	
	def forward(self, ori_input):
		for i in range(self.n_scales):
			# possibly downsample ori input
			if i!=self.n_scales-1:
				scale = 2**(self.n_scales-1-i)
				scaled_ori_input = F.interpolate(ori_input,scale_factor=1/scale, mode='bilinear', align_corners=True)
			else:
				scaled_ori_input = ori_input
			# conv layer
			if i==0:
				output = getattr(self, 'conv'+str(i))(scaled_ori_input)
			else:
				output = getattr(self, 'conv'+str(i))( torch.cat([scaled_ori_input,output], dim=1) )
			# possibly upsample output 
			if i!=self.n_scales-1:
				output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=True)
		return output


class B2SNet(nn.Module):
	def __init__(self, args=None):
		super(B2SNet, self).__init__()
		self.args=args
		assert self.args.mode == 'xx2x'
		self.n_scales = args.n_scales
		in_dim = 6
		med_dim = 64
		out_dim = 64
		for ind in range(self.n_scales):
			indim = in_dim if ind == 0 else in_dim+med_dim
			outdim = out_dim if ind == self.n_scales-1 else med_dim
			setattr(self, 'conv'+str(ind),
						B2SConvModule(indim, med_dim, outdim, 2, n_conv=2)
					)

		self.out_conv = nn.Sequential(
				nn.Conv2d(64, 32, 3, 1, 1),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(32, 3, 3, 1, 1)
			)



	def forward(self, input, mask=None, gt=None):
		ori_input = input[:, :6]
		for i in range(self.n_scales):
			# possibly downsample ori input
			if i!=self.n_scales-1:
				scale = 2**(self.n_scales-1-i)
				scaled_ori_input = F.interpolate(ori_input,scale_factor=1/scale, mode='bilinear', align_corners=True)
			else:
				scaled_ori_input = ori_input
			# conv layer
			if i==0:
				output = getattr(self, 'conv'+str(i))(scaled_ori_input)
			else:
				output = getattr(self, 'conv'+str(i))( torch.cat([scaled_ori_input,output], dim=1) )
			# possibly upsample output 
			if i!=self.n_scales-1:
				output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=True)	
		output = self.out_conv(output)
		# if self.args.runner in ['gen', 'refine' ]:
		return output, None