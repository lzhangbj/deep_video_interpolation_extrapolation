import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import numpy as np

BN_MOMENTUM = 0.01

def conv3x3(in_planes, out_planes, stride=1):
	"""3x3 convolution with padding"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
					 padding=1, bias=False)


class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(inplanes, planes, stride)
		# self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
		self.relu = nn.LeakyReLU(0.2,inplace=False)
		self.conv2 = conv3x3(planes, planes)
		# self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		# out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		# out = self.bn2(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out = out + residual
		out = self.relu(out)

		return out


class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(Bottleneck, self).__init__()
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
		# self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
							   padding=1, bias=False)
		# self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
		self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
							   bias=False)
		# self.bn3 = nn.BatchNorm2d(planes * self.expansion,
		#                      momentum=BN_MOMENTUM)
		self.relu = nn.LeakyReLU(0.2,inplace=False)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		# out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		# out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		# out = self.bn3(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out = out + residual
		out = self.relu(out)

		return out


class HighResolutionModule(nn.Module):
	def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
				 num_channels, fuse_method, multi_scale_output=True):
		super(HighResolutionModule, self).__init__()
		self._check_branches(
			num_branches, blocks, num_blocks, num_inchannels, num_channels)

		self.num_inchannels = num_inchannels
		self.fuse_method = fuse_method
		self.num_branches = num_branches

		self.multi_scale_output = multi_scale_output

		self.branches = self._make_branches(
			num_branches, blocks, num_blocks, num_channels)
		self.fuse_layers = self._make_fuse_layers()
		self.relu = nn.LeakyReLU(0.2,inplace=False)

	def _check_branches(self, num_branches, blocks, num_blocks, num_inchannels, num_channels):
		if num_branches != len(num_blocks):
			error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
				num_branches, len(num_blocks))
			raise ValueError(error_msg)

		if num_branches != len(num_channels):
			error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
				num_branches, len(num_channels))
			raise ValueError(error_msg)

		if num_branches != len(num_inchannels):
			error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
				num_branches, len(num_inchannels))
			raise ValueError(error_msg)

	def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
		downsample = None
		if stride != 1 or \
		   self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.num_inchannels[branch_index],
						  num_channels[branch_index] * block.expansion,
						  kernel_size=1, stride=stride, bias=False),
				# nn.BatchNorm2d(num_channels[branch_index] * block.expansion,
				#           momentum=BN_MOMENTUM),
			)

		layers = []
		layers.append(block(self.num_inchannels[branch_index],
							num_channels[branch_index], stride, downsample))
		self.num_inchannels[branch_index] = \
			num_channels[branch_index] * block.expansion
		for i in range(1, num_blocks[branch_index]):
			layers.append(block(self.num_inchannels[branch_index],
								num_channels[branch_index]))

		return nn.Sequential(*layers)

	def _make_branches(self, num_branches, block, num_blocks, num_channels):
		branches = []

		for i in range(num_branches):
			branches.append(
				self._make_one_branch(i, block, num_blocks, num_channels))

		return nn.ModuleList(branches)

	def _make_fuse_layers(self):
		if self.num_branches == 1:
			return None

		num_branches = self.num_branches
		num_inchannels = self.num_inchannels
		fuse_layers = []
		for i in range(num_branches if self.multi_scale_output else 1):
			fuse_layer = []
			for j in range(num_branches):
				if j > i:
					fuse_layer.append(nn.Sequential(
						nn.Conv2d(num_inchannels[j],
								  num_inchannels[i],
								  1,
								  1,
								  0,
								  bias=False)))
						# nn.BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM)))
				elif j == i:
					fuse_layer.append(None)
				else:
					conv3x3s = []
					for k in range(i-j):
						if k == i - j - 1:
							num_outchannels_conv3x3 = num_inchannels[i]
							conv3x3s.append(nn.Sequential(
								nn.Conv2d(num_inchannels[j],
										  num_outchannels_conv3x3,
										  3, 2, 1, bias=False)))
								# nn.BatchNorm2d(num_outchannels_conv3x3, 
								#           momentum=BN_MOMENTUM)))
						else:
							num_outchannels_conv3x3 = num_inchannels[j]
							conv3x3s.append(nn.Sequential(
								nn.Conv2d(num_inchannels[j],
										  num_outchannels_conv3x3,
										  3, 2, 1, bias=False),
								# nn.BatchNorm2d(num_outchannels_conv3x3,
								#           momentum=BN_MOMENTUM),
								nn.LeakyReLU(0.2,inplace=False)))
					fuse_layer.append(nn.Sequential(*conv3x3s))
			fuse_layers.append(nn.ModuleList(fuse_layer))

		return nn.ModuleList(fuse_layers)

	def get_num_inchannels(self):
		return self.num_inchannels

	def forward(self, x):
		if self.num_branches == 1:
			return [self.branches[0](x[0])]

		for i in range(self.num_branches):
			x[i] = self.branches[i](x[i])

		x_fuse = []
		for i in range(len(self.fuse_layers)):
			y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
			for j in range(1, self.num_branches):
				if i == j:
					y = y + x[j]
				elif j > i:
					width_output = x[i].shape[-1]
					height_output = x[i].shape[-2]
					y = y + F.interpolate(
						self.fuse_layers[i][j](x[j]),
						size=[height_output, width_output],
						mode='bilinear')
				else:
					y = y + self.fuse_layers[i][j](x[j])
			x_fuse.append(self.relu(y))

		return x_fuse


blocks_dict = {
	'BASIC': BasicBlock,
	'BOTTLENECK': Bottleneck
}


from yacs.config import CfgNode as CN

# high_resoluton_net related params for segmentation
HIGH_RESOLUTION_NET = CN()
HIGH_RESOLUTION_NET.PRETRAINED_LAYERS = ['*']
HIGH_RESOLUTION_NET.STEM_INPLANES = 64
HIGH_RESOLUTION_NET.FINAL_CONV_KERNEL = 1
HIGH_RESOLUTION_NET.WITH_HEAD = True

HIGH_RESOLUTION_NET.STAGE2 = CN()
HIGH_RESOLUTION_NET.STAGE2.NUM_MODULES = 1
HIGH_RESOLUTION_NET.STAGE2.NUM_BRANCHES = 2
HIGH_RESOLUTION_NET.STAGE2.NUM_BLOCKS = [4, 4]
HIGH_RESOLUTION_NET.STAGE2.NUM_CHANNELS = [64, 128]
HIGH_RESOLUTION_NET.STAGE2.BLOCK = 'BASIC'
HIGH_RESOLUTION_NET.STAGE2.FUSE_METHOD = 'SUM'


HIGH_RESOLUTION_NET.STAGE3 = CN()
HIGH_RESOLUTION_NET.STAGE3.NUM_MODULES = 1
HIGH_RESOLUTION_NET.STAGE3.NUM_BRANCHES = 3
HIGH_RESOLUTION_NET.STAGE3.NUM_BLOCKS = [4, 4, 4]
HIGH_RESOLUTION_NET.STAGE3.NUM_CHANNELS = [64, 128, 256]
HIGH_RESOLUTION_NET.STAGE3.BLOCK = 'BASIC'
HIGH_RESOLUTION_NET.STAGE3.FUSE_METHOD = 'SUM'





# high_resoluton_net related params for segmentation
# HIGH4_RESOLUTION_NET = CN()
# HIGH4_RESOLUTION_NET.PRETRAINED_LAYERS = ['*']
# HIGH4_RESOLUTION_NET.STEM_INPLANES = 64
# HIGH4_RESOLUTION_NET.FINAL_CONV_KERNEL = 1
# HIGH4_RESOLUTION_NET.WITH_HEAD = True

# HIGH4_RESOLUTION_NET.STAGE2 = CN()
# HIGH4_RESOLUTION_NET.STAGE2.NUM_MODULES = 1
# HIGH4_RESOLUTION_NET.STAGE2.NUM_BRANCHES = 2
# HIGH4_RESOLUTION_NET.STAGE2.NUM_BLOCKS = [4, 4]
# HIGH4_RESOLUTION_NET.STAGE2.NUM_CHANNELS = [128, 192]
# HIGH4_RESOLUTION_NET.STAGE2.BLOCK = 'BASIC'
# HIGH4_RESOLUTION_NET.STAGE2.FUSE_METHOD = 'SUM'


# HIGH4_RESOLUTION_NET.STAGE3 = CN()
# HIGH4_RESOLUTION_NET.STAGE3.NUM_MODULES = 1
# HIGH4_RESOLUTION_NET.STAGE3.NUM_BRANCHES = 3
# HIGH4_RESOLUTION_NET.STAGE3.NUM_BLOCKS = [4, 4, 4]
# HIGH4_RESOLUTION_NET.STAGE3.NUM_CHANNELS = [128, 192, 256]
# HIGH4_RESOLUTION_NET.STAGE3.BLOCK = 'BASIC'
# HIGH4_RESOLUTION_NET.STAGE3.FUSE_METHOD = 'SUM'


# HIGH4_RESOLUTION_NET.STAGE4 = CN()
# HIGH4_RESOLUTION_NET.STAGE4.NUM_MODULES = 1
# HIGH4_RESOLUTION_NET.STAGE4.NUM_BRANCHES = 4
# HIGH4_RESOLUTION_NET.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
# HIGH4_RESOLUTION_NET.STAGE4.NUM_CHANNELS = [128, 192, 256, 384]
# HIGH4_RESOLUTION_NET.STAGE4.BLOCK = 'BASIC'
# HIGH4_RESOLUTION_NET.STAGE4.FUSE_METHOD = 'SUM'

# high_resoluton_net related params for segmentation
HIGH4_RESOLUTION_NET = CN()
HIGH4_RESOLUTION_NET.PRETRAINED_LAYERS = ['*']
HIGH4_RESOLUTION_NET.STEM_INPLANES = 64
HIGH4_RESOLUTION_NET.FINAL_CONV_KERNEL = 1
HIGH4_RESOLUTION_NET.WITH_HEAD = True

HIGH4_RESOLUTION_NET.STAGE2 = CN()
HIGH4_RESOLUTION_NET.STAGE2.NUM_MODULES = 1
HIGH4_RESOLUTION_NET.STAGE2.NUM_BRANCHES = 2
HIGH4_RESOLUTION_NET.STAGE2.NUM_BLOCKS = [4, 4]
HIGH4_RESOLUTION_NET.STAGE2.NUM_CHANNELS = [64, 128]
HIGH4_RESOLUTION_NET.STAGE2.BLOCK = 'BASIC'
HIGH4_RESOLUTION_NET.STAGE2.FUSE_METHOD = 'SUM'


HIGH4_RESOLUTION_NET.STAGE3 = CN()
HIGH4_RESOLUTION_NET.STAGE3.NUM_MODULES = 1
HIGH4_RESOLUTION_NET.STAGE3.NUM_BRANCHES = 3
HIGH4_RESOLUTION_NET.STAGE3.NUM_BLOCKS = [4, 4, 4]
HIGH4_RESOLUTION_NET.STAGE3.NUM_CHANNELS = [64, 128, 256]
HIGH4_RESOLUTION_NET.STAGE3.BLOCK = 'BASIC'
HIGH4_RESOLUTION_NET.STAGE3.FUSE_METHOD = 'SUM'


HIGH4_RESOLUTION_NET.STAGE4 = CN()
HIGH4_RESOLUTION_NET.STAGE4.NUM_MODULES = 1
HIGH4_RESOLUTION_NET.STAGE4.NUM_BRANCHES = 4
HIGH4_RESOLUTION_NET.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
HIGH4_RESOLUTION_NET.STAGE4.NUM_CHANNELS = [64, 128, 256, 512]
HIGH4_RESOLUTION_NET.STAGE4.BLOCK = 'BASIC'
HIGH4_RESOLUTION_NET.STAGE4.FUSE_METHOD = 'SUM'

MODEL_EXTRAS = {
	'seg_hrnet': HIGH_RESOLUTION_NET,
}


has_stage4 = False

class HRNet(nn.Module):

	def __init__(self, args):
		super(HRNet, self).__init__()
		self.args = args
		extra = HIGH4_RESOLUTION_NET if self.args.highres_large else HIGH_RESOLUTION_NET
		self.seg_encode_dim=4
		self.n_classes=20
		if self.args.syn_type == 'extra':
			self.rgb_out_dim = 3*self.args.num_pred_once if not self.args.inpaint_mask else 4*self.args.num_pred_once 
		else:
			self.rgb_out_dim = 3
		# experiment shows it can not produce effective occlusion masks under no supervision
		self.seg_out_dim = 20*self.args.num_pred_once if self.args.syn_type == 'extra' else 20
		if self.args.syn_type == 'extra' and self.args.fix_init_frames:
			self.in_channel = (3+self.seg_encode_dim)*3
		else:
			self.in_channel = (3+self.seg_encode_dim)*2

		self.seg_encoder = nn.Sequential(
			nn.Conv2d(self.n_classes, 32, 3, 1, 1),
			nn.ELU(),
			nn.Conv2d(32, 32, 3,1,1),
			nn.ELU(),
			nn.Conv2d(32, self.seg_encode_dim, 3, 1, 1) 
		)

		# stem net
		self.conv1 = nn.Conv2d(self.in_channel, 64, kernel_size=3, stride=1, padding=1,
							   bias=True)
		# self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
		self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
							   bias=True)
		# self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
		self.relu = nn.LeakyReLU(0.2,inplace=False)

		self.layer1 = self._make_layer(Bottleneck, 64, 64, 4)

		self.stage2_cfg = extra['STAGE2']
		num_channels = self.stage2_cfg['NUM_CHANNELS']
		block = blocks_dict[self.stage2_cfg['BLOCK']]
		num_channels = [
			num_channels[i] * block.expansion for i in range(len(num_channels))]
		self.transition1 = self._make_transition_layer([256], num_channels)
		self.stage2, pre_stage_channels = self._make_stage(
			self.stage2_cfg, num_channels)

		self.stage3_cfg = extra['STAGE3']
		num_channels = self.stage3_cfg['NUM_CHANNELS']
		block = blocks_dict[self.stage3_cfg['BLOCK']]
		num_channels = [
			num_channels[i] * block.expansion for i in range(len(num_channels))]
		self.transition2 = self._make_transition_layer(
			pre_stage_channels, num_channels)
		self.stage3, pre_stage_channels = self._make_stage(
			self.stage3_cfg, num_channels)

		if self.args.highres_large:
			self.stage4_cfg = extra['STAGE4']
			num_channels = self.stage4_cfg['NUM_CHANNELS']
			block = blocks_dict[self.stage4_cfg['BLOCK']]
			num_channels = [
				num_channels[i] * block.expansion for i in range(len(num_channels))]
			self.transition3 = self._make_transition_layer(
				pre_stage_channels, num_channels)
			self.stage4, pre_stage_channels = self._make_stage(
				self.stage4_cfg, num_channels, multi_scale_output=True)
		
		last_inp_channels = np.int(np.sum(pre_stage_channels))


		self.rgb_layer = nn.Sequential(
				nn.Conv2d(
					in_channels=last_inp_channels,
					out_channels=last_inp_channels,
					kernel_size=1,
					stride=1,
					padding=0),
				# nn.BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
				nn.LeakyReLU(0.2,inplace=False),
				nn.Conv2d(
					in_channels=last_inp_channels,
					out_channels = self.rgb_out_dim,
					kernel_size=3,
					stride=1,
					padding=1)
			)

		self.seg_layer = nn.Sequential(
				nn.Conv2d(
					in_channels=last_inp_channels,
					out_channels=last_inp_channels,
					kernel_size=1,
					stride=1,
					padding=0),
				# nn.BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
				nn.LeakyReLU(0.2,inplace=False),
				nn.Conv2d(
					in_channels=last_inp_channels,
					out_channels = self.seg_out_dim,
					kernel_size=3,
					stride=1,
					padding=1)
			)

	def _make_transition_layer( self, num_channels_pre_layer, num_channels_cur_layer):
		num_branches_cur = len(num_channels_cur_layer)
		num_branches_pre = len(num_channels_pre_layer)

		transition_layers = []
		for i in range(num_branches_cur):
			if i < num_branches_pre:
				if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
					transition_layers.append(nn.Sequential(
						nn.Conv2d(num_channels_pre_layer[i],
								  num_channels_cur_layer[i],
								  3,
								  1,
								  1,
								  bias=False),
						# nn.BatchNorm2d(
						#   num_channels_cur_layer[i], momentum=BN_MOMENTUM),
						nn.LeakyReLU(0.2,inplace=False)))
				else:
					transition_layers.append(None)
			else:
				conv3x3s = []
				for j in range(i+1-num_branches_pre):
					inchannels = num_channels_pre_layer[-1]
					outchannels = num_channels_cur_layer[i] \
						if j == i-num_branches_pre else inchannels
					conv3x3s.append(nn.Sequential(
						nn.Conv2d(
							inchannels, outchannels, 3, 2, 1, bias=False),
						# nn.BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
						nn.LeakyReLU(0.2,inplace=False)))
				transition_layers.append(nn.Sequential(*conv3x3s))

		return nn.ModuleList(transition_layers)

	def _make_layer(self, block, inplanes, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(inplanes, planes * block.expansion,
						  kernel_size=1, stride=stride, bias=False),
				# nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
			)

		layers = []
		layers.append(block(inplanes, planes, stride, downsample))
		inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(inplanes, planes))

		return nn.Sequential(*layers)

	def _make_stage(self, layer_config, num_inchannels,multi_scale_output=True):
		num_modules = layer_config['NUM_MODULES']
		num_branches = layer_config['NUM_BRANCHES']
		num_blocks = layer_config['NUM_BLOCKS']
		num_channels = layer_config['NUM_CHANNELS']
		block = blocks_dict[layer_config['BLOCK']]
		fuse_method = layer_config['FUSE_METHOD']

		modules = []
		for i in range(num_modules):
			# multi_scale_output is only used last module
			if not multi_scale_output and i == num_modules - 1:
				reset_multi_scale_output = False
			else:
				reset_multi_scale_output = True
			modules.append(
				HighResolutionModule(num_branches,
									  block,
									  num_blocks,
									  num_inchannels,
									  num_channels,
									  fuse_method,
									  reset_multi_scale_output)
			)
			num_inchannels = modules[-1].get_num_inchannels()

		return nn.Sequential(*modules), num_inchannels

	def forward(self, input):
		if self.args.syn_type == 'extra' and self.args.fix_init_frames:
			encoded_segs = [
				self.seg_encoder(input[:, 9:9+self.n_classes]),
				self.seg_encoder(input[:, 9+self.n_classes:9+2*self.n_classes]),
				self.seg_encoder(input[:, 9+2*self.n_classes:9+3*self.n_classes])
			]
			encoded_feat = torch.cat([input[:,:9]] + encoded_segs, dim=1)
		else:
			encoded_segs = [
				self.seg_encoder(input[:, 6:6+self.n_classes]),
				self.seg_encoder(input[:, 6+self.n_classes:6+2*self.n_classes])
			]
			encoded_feat = torch.cat([input[:,:6]] + encoded_segs, dim=1)

		x = encoded_feat

		x = self.conv1(x)
		x = self.relu(x)
		x = self.conv2(x)
		x = self.relu(x)
		x = self.layer1(x)

		x_list = []
		for i in range(self.stage2_cfg['NUM_BRANCHES']):
			if self.transition1[i] is not None:
				x_list.append(self.transition1[i](x))
			else:
				x_list.append(x)
		y_list = self.stage2(x_list)

		x_list = []
		for i in range(self.stage3_cfg['NUM_BRANCHES']):
			if self.transition2[i] is not None:
				x_list.append(self.transition2[i](y_list[-1]))
			else:
				x_list.append(y_list[i])
		if self.args.highres_large:
			y_list = self.stage3(x_list)
		else:
			x = self.stage3(x_list)

		if self.args.highres_large:
			x_list = []
			for i in range(self.stage4_cfg['NUM_BRANCHES']):
				if self.transition3[i] is not None:
					x_list.append(self.transition3[i](y_list[-1]))
				else:
					x_list.append(y_list[i])
			x = self.stage4(x_list)

		# Upsampling
		x0_h, x0_w = x[0].size(2), x[0].size(3)
		x1 = F.upsample(x[1], size=(x0_h, x0_w), mode='bilinear')
		x2 = F.upsample(x[2], size=(x0_h, x0_w), mode='bilinear')
		if self.args.highres_large:
			x3 = F.upsample(x[3], size=(x0_h, x0_w), mode='bilinear')

		x = torch.cat([x[0], x1, x2, x3], 1) if self.args.highres_large else torch.cat([x[0], x1, x2], 1)

		rgb_out = self.rgb_layer(x)
		if self.args.syn_type == 'extra' and self.args.inpaint and self.args.inpaint_mask:
			mask_out = F.sigmoid(rgb_out[:, 3*self.args.num_pred_once:])
		rgb_out = rgb_out[:, :3*self.args.num_pred_once] if self.args.syn_type == 'extra' else rgb_out
		seg_out = self.seg_layer(x)

		if self.args.syn_type == 'extra' and self.args.inpaint and not self.args.inpaint_mask:
			# produce mask by segmentation
			seg_input    = torch.argmax(input[:, -20:], dim=1, keepdim=True)
			seg_int_list = [ torch.argmax(seg_out[:,20*i:20*i+20], dim=1, keepdim=True).detach() for i in range(self.args.num_pred_once)]
			foreground_input = (seg_input>=11).float()
			background_pred = [(s<11).float() for s in seg_int_list]
			mask_out = torch.cat([1-(foreground_input*s).float() for s in background_pred], dim=1)

		if self.args.syn_type == 'extra' and self.args.inpaint:
			return rgb_out, seg_out, mask_out
		else:
			return rgb_out, seg_out


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


class InpaintUnet(nn.Module):
	def __init__(self, args):
		super(InpaintUnet, self).__init__()
		self.args=args
		self.in_dim = (3+1+20)*self.args.num_pred_once
		self.out_dim = 3*self.args.num_pred_once

		self.input_trans = nn.Sequential(
				nn.Conv2d(self.in_dim, 128, 5, 1, 2),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(128, 64, 5, 1, 2),
				nn.LeakyReLU(0.2, inplace=True)
			)

		self.encoder_1 = nn.Sequential(             # out 128, x0.5
				nn.Conv2d(64, 128, 3, 2, 1),
				nn.LeakyReLU(0.2, inplace=True),
				ResnetBlock(128, 128, 3)
			)                           

		self.encoder_2 = nn.Sequential(             # out 256, x0.25
				nn.Conv2d(128, 256, 3, 2, 1),
				nn.LeakyReLU(0.2, inplace=True),
				ResnetBlock(256, 256, 3)
			)       

		self.bottle_dilated = nn.Sequential(        # out 256, x0.25
				nn.Conv2d(256, 256, 3, 1, 1, 1),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(256, 256, 3, 1, 2, 2),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(256, 256, 3, 1, 4, 4),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(256, 256, 3, 1, 8, 8)
			)   

		self.decoder_2 = nn.Sequential(             # in + skip, out 128, x0.5
				ResnetBlock(256, 256, 3),
				nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(256, 128, 3, 1, 1)
			)   

		self.decoder_1 = nn.Sequential(             # in + skip, out 64, x1
				ResnetBlock(128, 128, 3),
				nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(128, 64, 3, 1, 1)
			)

		self.out = nn.Sequential(
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(64, 64, 3, 1, 1),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(64, self.out_dim, 3, 1, 1)
			)

	def forward(self, rgb, mask, seg):
		input = torch.cat([rgb, mask, seg], dim=1)
		input_transed = self.input_trans(input)

		encoded_1 = self.encoder_1(input_transed)
		encoded_2 = self.encoder_2(encoded_1)

		dilated_out = self.bottle_dilated(encoded_2)

		decoded_2 = self.decoder_2(encoded_2 + dilated_out)
		decoded_1 = self.decoder_1(decoded_2 + encoded_1)

		out = self.out(decoded_1 + input_transed)#.clamp_(-1,1)
		# out.clamp_(-1,1)

		out_list  = [ out[:, 3*j:3*j+3] * (1-mask[:, j:j+1])  for j in range(self.args.num_pred_once) ]

		rgb_list  = [ rgb[:, 3*j:3*j+3] * mask[:, j:j+1]  for j in range(self.args.num_pred_once) ]

		final_list = []
		for i in range(self.args.num_pred_once):
			final_list.append(out_list[i] + rgb_list[i])


		return torch.cat(final_list, dim=1)


class VAEHRNet(nn.Module):

	def __init__(self, args):
		super(VAEHRNet, self).__init__()
		self.args = args
		extra = HIGH4_RESOLUTION_NET if self.args.highres_large else HIGH_RESOLUTION_NET
		self.seg_encode_dim=4
		self.vae_channel=32
		self.n_classes=20
		if self.args.syn_type == 'extra':
			self.rgb_out_dim = 3*self.args.num_pred_once if not self.args.inpaint_mask else 4*self.args.num_pred_once 
		else:
			self.rgb_out_dim = 3
		# experiment shows it can not produce effective occlusion masks under no supervision
		self.seg_out_dim = 20*self.args.num_pred_once if self.args.syn_type == 'extra' else 20
		if self.args.syn_type == 'extra' and self.args.fix_init_frames:
			self.in_channel = (3+self.seg_encode_dim)*3 + self.vae_channel
		else:
			self.in_channel = (3+self.seg_encode_dim)*2 + self.vae_channel

		self.vae_encoder = nn.Sequential(   
				nn.Conv2d(23*3, 32, 3,1,1),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(32, 32, 3,1,1),
				nn.BatchNorm2d(32),
				nn.LeakyReLU(0.2, inplace=True),

				# downsize 1  32*64*64
				nn.Conv2d(32, 32, 3, 2, 1),
				nn.BatchNorm2d(32),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(32, 32, 3, 1, 1),
				nn.BatchNorm2d(32),
				nn.LeakyReLU(0.2, inplace=True),
				# downsize 2  64*32*32
				nn.Conv2d(32, 64, 3, 2, 1),
				nn.BatchNorm2d(64),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(64, 64, 3, 1, 1),
				nn.BatchNorm2d(64),
				nn.LeakyReLU(0.2, inplace=True),
				# downsize 3 128*16*16
				nn.Conv2d(64, 128, 3, 2, 1),
				nn.BatchNorm2d(128),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(128, 128, 3, 1, 1),
				nn.BatchNorm2d(128),
				nn.LeakyReLU(0.2, inplace=True),
				# downsize 4 128*8*8 --> 16*8*8
				nn.Conv2d(128, 128, 3, 2, 1),
				nn.BatchNorm2d(128),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(128, 64, 3, 1, 1),
				nn.BatchNorm2d(64),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(64, 32, 3, 1, 1),
				nn.BatchNorm2d(32),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(32, 16, 3, 1, 1)
			)
		self.mu_fc = nn.Linear(1024, 1024)
		self.logvar_fc = nn.Linear(1024, 1024)
		self.vae_decoder = nn.Sequential(
				# upsample 1  32*16*16
				nn.ConvTranspose2d(16, 32, 4, stride=2, padding=1),
				nn.BatchNorm2d(32),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(32, 32, 3,1,1),
				nn.BatchNorm2d(32),
				nn.LeakyReLU(0.2, inplace=True),
				# upsample 2  32*32*32
				nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1),
				nn.BatchNorm2d(32),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(32, 32, 3,1,1),
				nn.BatchNorm2d(32),
				nn.LeakyReLU(0.2, inplace=True),
				# upsample 3  32*64*64
				nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1),
				nn.BatchNorm2d(32),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(32, 32, 3,1,1),
				nn.BatchNorm2d(32),
				nn.LeakyReLU(0.2, inplace=True),
				# upsample 4  32*128*128
				nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1),
				nn.BatchNorm2d(32),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Conv2d(32, 32, 3,1,1)    
			)


		self.seg_encoder = nn.Sequential(
			nn.Conv2d(self.n_classes, 32, 3, 1, 1),
			nn.ELU(),
			nn.Conv2d(32, 32, 3,1,1),
			nn.ELU(),
			nn.Conv2d(32, self.seg_encode_dim, 3, 1, 1) 
		)

		# stem net
		self.conv1 = nn.Conv2d(self.in_channel, 64, kernel_size=3, stride=1, padding=1,
							   bias=True)
		# self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
		self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
							   bias=True)
		# self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
		self.relu = nn.LeakyReLU(0.2,inplace=False)

		self.layer1 = self._make_layer(Bottleneck, 64, 64, 4)

		self.stage2_cfg = extra['STAGE2']
		num_channels = self.stage2_cfg['NUM_CHANNELS']
		block = blocks_dict[self.stage2_cfg['BLOCK']]
		num_channels = [
			num_channels[i] * block.expansion for i in range(len(num_channels))]
		self.transition1 = self._make_transition_layer([256], num_channels)
		self.stage2, pre_stage_channels = self._make_stage(
			self.stage2_cfg, num_channels)

		self.stage3_cfg = extra['STAGE3']
		num_channels = self.stage3_cfg['NUM_CHANNELS']
		block = blocks_dict[self.stage3_cfg['BLOCK']]
		num_channels = [
			num_channels[i] * block.expansion for i in range(len(num_channels))]
		self.transition2 = self._make_transition_layer(
			pre_stage_channels, num_channels)
		self.stage3, pre_stage_channels = self._make_stage(
			self.stage3_cfg, num_channels)

		if self.args.highres_large:
			self.stage4_cfg = extra['STAGE4']
			num_channels = self.stage4_cfg['NUM_CHANNELS']
			block = blocks_dict[self.stage4_cfg['BLOCK']]
			num_channels = [
				num_channels[i] * block.expansion for i in range(len(num_channels))]
			self.transition3 = self._make_transition_layer(
				pre_stage_channels, num_channels)
			self.stage4, pre_stage_channels = self._make_stage(
				self.stage4_cfg, num_channels, multi_scale_output=True)
		
		last_inp_channels = np.int(np.sum(pre_stage_channels))


		self.rgb_layer = nn.Sequential(
				nn.Conv2d(
					in_channels=last_inp_channels,
					out_channels=last_inp_channels,
					kernel_size=1,
					stride=1,
					padding=0),
				# nn.BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
				nn.LeakyReLU(0.2,inplace=False),
				nn.Conv2d(
					in_channels=last_inp_channels,
					out_channels = self.rgb_out_dim,
					kernel_size=3,
					stride=1,
					padding=1)
			)

		self.seg_layer = nn.Sequential(
				nn.Conv2d(
					in_channels=last_inp_channels,
					out_channels=last_inp_channels,
					kernel_size=1,
					stride=1,
					padding=0),
				# nn.BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
				nn.LeakyReLU(0.2,inplace=False),
				nn.Conv2d(
					in_channels=last_inp_channels,
					out_channels = self.seg_out_dim,
					kernel_size=3,
					stride=1,
					padding=1)
			)

	def _make_transition_layer( self, num_channels_pre_layer, num_channels_cur_layer):
		num_branches_cur = len(num_channels_cur_layer)
		num_branches_pre = len(num_channels_pre_layer)

		transition_layers = []
		for i in range(num_branches_cur):
			if i < num_branches_pre:
				if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
					transition_layers.append(nn.Sequential(
						nn.Conv2d(num_channels_pre_layer[i],
								  num_channels_cur_layer[i],
								  3,
								  1,
								  1,
								  bias=False),
						# nn.BatchNorm2d(
						#   num_channels_cur_layer[i], momentum=BN_MOMENTUM),
						nn.LeakyReLU(0.2,inplace=False)))
				else:
					transition_layers.append(None)
			else:
				conv3x3s = []
				for j in range(i+1-num_branches_pre):
					inchannels = num_channels_pre_layer[-1]
					outchannels = num_channels_cur_layer[i] \
						if j == i-num_branches_pre else inchannels
					conv3x3s.append(nn.Sequential(
						nn.Conv2d(
							inchannels, outchannels, 3, 2, 1, bias=False),
						# nn.BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
						nn.LeakyReLU(0.2,inplace=False)))
				transition_layers.append(nn.Sequential(*conv3x3s))

		return nn.ModuleList(transition_layers)

	def _make_layer(self, block, inplanes, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(inplanes, planes * block.expansion,
						  kernel_size=1, stride=stride, bias=False),
				# nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
			)

		layers = []
		layers.append(block(inplanes, planes, stride, downsample))
		inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(inplanes, planes))

		return nn.Sequential(*layers)

	def _make_stage(self, layer_config, num_inchannels,multi_scale_output=True):
		num_modules = layer_config['NUM_MODULES']
		num_branches = layer_config['NUM_BRANCHES']
		num_blocks = layer_config['NUM_BLOCKS']
		num_channels = layer_config['NUM_CHANNELS']
		block = blocks_dict[layer_config['BLOCK']]
		fuse_method = layer_config['FUSE_METHOD']

		modules = []
		for i in range(num_modules):
			# multi_scale_output is only used last module
			if not multi_scale_output and i == num_modules - 1:
				reset_multi_scale_output = False
			else:
				reset_multi_scale_output = True
			modules.append(
				HighResolutionModule(num_branches,
									  block,
									  num_blocks,
									  num_inchannels,
									  num_channels,
									  fuse_method,
									  reset_multi_scale_output)
			)
			num_inchannels = modules[-1].get_num_inchannels()

		return nn.Sequential(*modules), num_inchannels

	def reparameterize(self, mu=None, logvar=None, bs=None):
		if self.training:
			std = logvar.mul(0.5).exp_()
			eps = std.new(std.size()).normal_()
			return eps.mul(std).add_(mu)
		else:
			return torch.zeros(bs, 1024).normal_().cuda(self.args.rank)

	def forward(self, input, gt_x=None, gt_seg=None):
		mu=None
		logvar=None
		if self.training:
			vae_input= torch.cat([input, gt_x, gt_seg], dim=1)
			vae_encoded = self.vae_encoder(vae_input)
			vae_encoded = vae_encoded.view(-1, 1024)
			mu = self.mu_fc(vae_encoded)
			logvar = self.logvar_fc(vae_encoded)
			z = self.reparameterize(mu, logvar)
		else:
			z = self.reparameterize(bs=input.size(0))

		z = z.view(-1, 16, 8, 8)
		vae_feature = self.vae_decoder(z)


		if self.args.syn_type == 'extra' and self.args.fix_init_frames:
			encoded_segs = [
				self.seg_encoder(input[:, 9:9+self.n_classes]),
				self.seg_encoder(input[:, 9+self.n_classes:9+2*self.n_classes]),
				self.seg_encoder(input[:, 9+2*self.n_classes:9+3*self.n_classes])
			]
			encoded_feat = torch.cat([vae_feature, input[:,:9]] + encoded_segs, dim=1)
		else:
			encoded_segs = [
				self.seg_encoder(input[:, 6:6+self.n_classes]),
				self.seg_encoder(input[:, 6+self.n_classes:6+2*self.n_classes])
			]
			encoded_feat = torch.cat([vae_feature, input[:,:6]] + encoded_segs, dim=1)

		x = encoded_feat

		x = self.conv1(x)
		x = self.relu(x)
		x = self.conv2(x)
		x = self.relu(x)
		x = self.layer1(x)

		x_list = []
		for i in range(self.stage2_cfg['NUM_BRANCHES']):
			if self.transition1[i] is not None:
				x_list.append(self.transition1[i](x))
			else:
				x_list.append(x)
		y_list = self.stage2(x_list)

		x_list = []
		for i in range(self.stage3_cfg['NUM_BRANCHES']):
			if self.transition2[i] is not None:
				x_list.append(self.transition2[i](y_list[-1]))
			else:
				x_list.append(y_list[i])
		if self.args.highres_large:
			y_list = self.stage3(x_list)
		else:
			x = self.stage3(x_list)

		if self.args.highres_large:
			x_list = []
			for i in range(self.stage4_cfg['NUM_BRANCHES']):
				if self.transition3[i] is not None:
					x_list.append(self.transition3[i](y_list[-1]))
				else:
					x_list.append(y_list[i])
			x = self.stage4(x_list)

		# Upsampling
		x0_h, x0_w = x[0].size(2), x[0].size(3)
		x1 = F.upsample(x[1], size=(x0_h, x0_w), mode='bilinear')
		x2 = F.upsample(x[2], size=(x0_h, x0_w), mode='bilinear')
		if self.args.highres_large:
			x3 = F.upsample(x[3], size=(x0_h, x0_w), mode='bilinear')

		x = torch.cat([x[0], x1, x2, x3], 1) if self.args.highres_large else torch.cat([x[0], x1, x2], 1)

		rgb_out = self.rgb_layer(x)
		if self.args.syn_type == 'extra' and self.args.inpaint and self.args.inpaint_mask:
			mask_out = F.sigmoid(rgb_out[:, 3*self.args.num_pred_once:])
		rgb_out = rgb_out[:, :3*self.args.num_pred_once] if self.args.syn_type == 'extra' else rgb_out
		seg_out = self.seg_layer(x)

		if self.args.syn_type == 'extra' and self.args.inpaint and not self.args.inpaint_mask:
			# produce mask by segmentation
			seg_input    = torch.argmax(input[:, -20:], dim=1, keepdim=True)
			seg_int_list = [ torch.argmax(seg_out[:,20*i:20*i+20], dim=1, keepdim=True).detach() for i in range(self.args.num_pred_once)]
			foreground_input = (seg_input>=11).float()
			background_pred = [(s<11).float() for s in seg_int_list]
			mask_out = torch.cat([1-(foreground_input*s).float() for s in background_pred], dim=1)

		if self.args.syn_type == 'extra' and self.args.inpaint:
			return rgb_out, seg_out, mask_out
		else:
			return rgb_out, seg_out, mu, logvar

