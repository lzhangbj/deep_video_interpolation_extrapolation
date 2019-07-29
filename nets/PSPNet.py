import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

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


class PSPConv(nn.Module):
	def __init__(self, in_dim, out_dim, n_scales, layers, inter_dims, kernel_sizes):
		super(PSPConv, self).__init__()
		self.n_scales = n_scales
		self.scales = [2**i for i in range(n_scales)] # 1,2,4,...
		assert len(inter_dims) == n_scales
		assert len(kernel_sizes) == n_scales

		for i in range(self.n_scales):
			seq = []
			indim  = in_dim
			outdim = inter_dims[i]
			for t in range(i):
				seq+=[	nn.Conv2d(indim, outdim, 3, 2, 1),
						nn.LeakyReLU(0.2, inplace=True) ]
				indim=outdim
			for j in range(layers):
				seq.append(ResnetBlock(inter_dims[i], inter_dims[i], kernel_sizes[i]))
			setattr(self, 'scale_'+str(i)+"_conv", nn.Sequential(*seq))

		self.tail_conv = nn.Sequential(
							nn.LeakyReLU(0.2, inplace=True),
							nn.Conv2d(sum(inter_dims), out_dim, 3, 1, 1)
						)

	def forward(self, input):
		outs = []
		for i in range(self.n_scales):
			out = getattr(self, 'scale_'+str(i)+"_conv")(input)
			if i>0:
				out = F.interpolate(out, scale_factor=self.scales[i], mode='bilinear', align_corners=True)
			outs.append(out)
		out = torch.cat(outs, dim=1)
		out = self.tail_conv(out)
		return out


class PSPNet(nn.Module):
	def __init__(self, args=None):
		super(PSPNet, self).__init__()
		self.n_classes = 20 
		self.seg_encode_dim = 4
		self.args=args
		if self.args.mode == 'xs2xs':
			self.in_channel = (3+self.seg_encode_dim)*2
		elif self.args.mode =='xx2x':
			self.in_channel=6
		self.n_channels = [64,128,256]

		if self.args.mode == 'xs2xs':
			self.seg_encoder = nn.Sequential(
				nn.Conv2d(self.n_classes, 32, 3, 1, 1),
				nn.ELU(),
				nn.Conv2d(32, 32, 3,1,1),
				nn.ELU(),
				nn.Conv2d(32, self.seg_encode_dim, 3, 1, 1) 
			)
			self.seg_tail = nn.Sequential(
				nn.LeakyReLU(0.2),
				nn.Conv2d(64, 64, 3, 1, 1),
				nn.LeakyReLU(0.2),
				nn.Conv2d(64, 20, 3, 1, 1)
			)

		self.head_conv = nn.Sequential(
				nn.Conv2d(self.in_channel, 64, 3, 1, 1),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(64, 64, 3, 1, 1)
			)

		for i in range(4):
			setattr(self, 'pspconv_'+str(i), PSPConv(64, 64, 3, 2, self.n_channels, [3,3,3]))

		self.rgb_tail = nn.Sequential(
				nn.LeakyReLU(0.2),
				nn.Conv2d(64, 32, 3, 1, 1),
				nn.LeakyReLU(0.2),
				nn.Conv2d(32, 3, 3, 1, 1)
			)


	def forward(self, input, mask=None, gt=None):
		if self.args.mode == 'xs2xs':
			encoded_segs = [
				self.seg_encoder(input[:, 6:6+self.n_classes]),
				self.seg_encoder(input[:, 6+self.n_classes:6+2*self.n_classes])
			]

		if self.args.mode == 'xs2xs':
			encoded_feat = torch.cat([input[:,:6]] + encoded_segs, dim=1)
		elif self.args.mode == 'xx2x':
			encoded_feat = input[:,:6]

		psp_input = self.head_conv(encoded_feat)

		for i in range(4):
			output = getattr(self, 'pspconv_'+str(i))(psp_input)
			psp_input=output

		output_rgb = self.rgb_tail(output)
		output_rgb.clamp_(-1,1)

		output_seg = None
		if self.args.mode == 'xs2xs':
			output_seg = self.seg_tail(output)
		if self.args.runner in ['gen', 'refine' ]:
			return output_rgb, output_seg
		else:
			return output_rgb,  output_seg, None



class PSPConvV2(nn.Module):
	def __init__(self, in_dim, out_dim, n_scales, layers, inter_dims, kernel_sizes):
		super(PSPConvV2, self).__init__()
		self.n_scales = n_scales
		self.scales = [2**i for i in range(n_scales)] # 1,2,4,...
		assert len(inter_dims) == n_scales
		assert len(kernel_sizes) == n_scales

		for i in range(self.n_scales):
			seq = []
			indim  = in_dim
			outdim = inter_dims[i]
			for t in range(i):
				seq+=[	nn.Conv2d(indim, outdim, 3, 2, 1),
						nn.LeakyReLU(0.2) ]
				indim=outdim
			setattr(self, 'scale_'+str(i)+"_conv_head", nn.Sequential(*seq))
			seq=[]
			for j in range(layers):
				seq.append(ResnetBlock(inter_dims[i], inter_dims[i], kernel_sizes[i]))
			setattr(self, 'scale_'+str(i)+"_conv", nn.Sequential(*seq))

		self.tail_conv = nn.Sequential(
							nn.LeakyReLU(0.2),
							nn.Conv2d(sum(inter_dims), out_dim, 3, 1, 1)
						)

	def forward(self, input, feats=None):
		outs = []
		out_feats = []
		for i in range(self.n_scales):
			out = getattr(self, 'scale_'+str(i)+"_conv_head")(input)
			if feats is not None:
				out += feats[i]
			out = getattr(self, 'scale_'+str(i)+"_conv")(out)
			out_feats.append(out)
			if i>0:
				out = F.interpolate(out, scale_factor=self.scales[i], mode='bilinear', align_corners=True)
			outs.append(out)
		out = torch.cat(outs, dim=1)
		out = self.tail_conv(out)
		return out, out_feats


class PSPNetV2(nn.Module):
	def __init__(self, args=None):
		super(PSPNetV2, self).__init__()
		self.n_classes = 20 
		self.seg_encode_dim = 4
		self.args=args
		if self.args.mode == 'xs2xs':
			self.in_channel = (3+self.seg_encode_dim)*2
		elif self.args.mode =='xx2x':
			self.in_channel=6
		self.n_channels = [64,128,256]

		if self.args.mode == 'xs2xs':
			self.seg_encoder = nn.Sequential(
				nn.Conv2d(self.n_classes, 32, 3, 1, 1),
				nn.ELU(),
				nn.Conv2d(32, 32, 3,1,1),
				nn.ELU(),
				nn.Conv2d(32, self.seg_encode_dim, 3, 1, 1) 
			)
			self.seg_tail = nn.Sequential(
				nn.LeakyReLU(0.2),
				nn.Conv2d(64, 64, 3, 1, 1),
				nn.LeakyReLU(0.2),
				nn.Conv2d(64, 20, 3, 1, 1)
			)

		self.head_conv = nn.Sequential(
				nn.Conv2d(self.in_channel, 64, 3, 1, 1),
				nn.LeakyReLU(0.2),
				nn.Conv2d(64, 64, 3, 1, 1)
			)

		for i in range(4):
			setattr(self, 'pspconv_'+str(i), PSPConvV2(64, 64, 3, 2, self.n_channels, [3,3,3]))

		self.rgb_tail = nn.Sequential(
				nn.LeakyReLU(0.2),
				nn.Conv2d(64, 32, 3, 1, 1),
				nn.LeakyReLU(0.2),
				nn.Conv2d(32, 3, 3, 1, 1)
			)


	def forward(self, input, mask=None, gt=None):
		if self.args.mode == 'xs2xs':
			encoded_segs = [
				self.seg_encoder(input[:, 6:6+self.n_classes]),
				self.seg_encoder(input[:, 6+self.n_classes:6+2*self.n_classes])
			]

		if self.args.mode == 'xs2xs':
			encoded_feat = torch.cat([input[:,:6]] + encoded_segs, dim=1)
		elif self.args.mode == 'xx2x':
			encoded_feat = input[:,:6]

		psp_input = self.head_conv(encoded_feat)
		out_features = None
		for i in range(4):
			output, out_features = getattr(self, 'pspconv_'+str(i))(psp_input, out_features)
			psp_input=output

		output_rgb = self.rgb_tail(output)
		output_rgb.clamp_(-1,1)

		output_seg = None
		if self.args.mode == 'xs2xs':
			output_seg = self.seg_tail(output)
		if self.args.runner in ['gen', 'refine' ]:
			return output_rgb, output_seg
		else:
			return output_rgb,  output_seg, None
