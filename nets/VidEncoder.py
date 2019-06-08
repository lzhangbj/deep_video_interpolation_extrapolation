import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from nets.unet2stream import *
# from nets.vgg import *
# from utils.net_utils import *
# from losses import RGBLoss

MODE_LIST = ['s2s', 'x2x', 'xs2s', 'xs2x']


class VidEncoder(nn.Module):
	def __init__(self, n_classes, seg_id=False, length=8, args=None):
		super(VidEncoder, self).__init__()