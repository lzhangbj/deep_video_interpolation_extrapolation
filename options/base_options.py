import argparse
import os
import torch

class BaseOptions():
	def __init__(self):
		self.parser = argparse.ArgumentParser()
		self.initialized = False

	def initialize(self):                
		self.parser.add_argument('--dataset', dest='dataset',
												default='cityscape',
												help='training dataset', 
												choices=['cityscape'])
		self.parser.add_argument('--runner', dest='runner',
												default=None,
												help='training dataset', 
												choices=['trainer', 'ganer'])
		self.parser.add_argument('--val', dest='val', 
												help='whether eval after each training ', 
												action='store_true')
		self.parser.add_argument('--val_interval', dest='val_interval',
												help='number of epochs to evaluate',
												type=int, 
												default=1)
		self.parser.add_argument('--img_dir', dest='img_dir',
												help='directory to load models', default=None,
												type=str)
		self.parser.add_argument('--seg_dir', dest='seg_dir',
												help='directory to load models', default=None,
												type=str)


		self.parser.add_argument('--syn_type', dest='syn_type',
												help='synthesize method',
												choices=['inter', 'extra'],
												default='extra') 
		self.parser.add_argument('--mode', dest='mode',
												help='mode to use',
												choices=['xs2xs', 'xss2x'],
												default='xs2xs')
		self.parser.add_argument('--bs', dest='batch_size', 
												type=int,
												default=1, 
												help='Batch size (over multiple gpu)')
		self.parser.add_argument('--epochs', dest='epochs', 
												type=int,
												default=20, 
												help='Number of training epochs')


		# weight of losses
		self.parser.add_argument('--l1_w', dest='l1_weight',
												help='training optimizer loss weigh of l1',
												type=float,
												default=20)
		self.parser.add_argument('--gdl_w', dest='gdl_weight',
												help='training optimizer loss weigh of gdl',
												type=float,
												default=20)
		self.parser.add_argument('--vgg_w', dest='vgg_weight',
												help='training optimizer loss weigh of vgg',
												type=float,
												default=20)
		self.parser.add_argument('--ce_w', dest='ce_weight',
												help='training optimizer loss weigh of vgg',
												type=float,
												default=10)     
		self.parser.add_argument('--ssim_w', dest='ssim_weight',
												help='training optimizer loss weigh of feat',
												type=float,
												default=10)     



		# distributed training
		self.parser.add_argument('--nw',  dest='num_workers', 
												type=int, default=4,
												help='Number of data loading workers')
		self.parser.add_argument('--port', dest='port',
												type=int, default=None, 
												help='Port for distributed training')
		self.parser.add_argument('--seed', type=int,
												default=1024, help='Random seed')

		self.parser.add_argument('--start_epoch', dest='start_epoch',
												help='starting epoch',
												default=1, type=int)
		self.parser.add_argument('--disp_interval', dest='disp_interval',
												help='number of iterations to display',
												default=10, type=int)
		# config optimization
		self.parser.add_argument('--lr_decay_step', dest='lr_decay_step', 
												help='step to do learning rate decay, unit is epoch',
												default=5, type=int)
		self.parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
												help='learning rate decay ratio', 
												default=1, type=float)
		self.parser.add_argument('--save_dir', dest='save_dir',
												help='directory to load models', default="log",
												type=str)


		# resume
		# set training session
		self.parser.add_argument('--s', dest='session',
												help='training session',
												default=1, type=int)

		# resume trained model
		self.parser.add_argument('--r', dest='resume',
												help='resume checkpoint or not',
												default=False, type=bool)
		self.parser.add_argument('--checksession', dest='checksession',
												help='checksession to load model',
												default=1, type=int)
		self.parser.add_argument('--checkepoch', dest='checkepoch',
												help='checkepoch to load model',
												default=1, type=int)
		self.parser.add_argument('--checkpoint', dest='checkpoint',
												help='checkpoint to load model',
												default=0, type=int)
		self.parser.add_argument('--load_dir', dest='load_dir',
												help='directory to load models', default="models",
												type=str)
		self.initialized = True


	def parse(self, save=True):
		if not self.initialized:
			self.initialize()
		self.opt = self.parser.parse_args()
		return self.opt
