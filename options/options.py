import argparse
import os
import torch

class Options():
	def __init__(self):
		self.parser = argparse.ArgumentParser()
		self.initialized = False

	def initialize(self):                
		self.parser.add_argument('--dataset', dest='dataset',
												default='cityscape',
												help='training dataset', 
												choices=['cityscape'])
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
		self.parser.add_argument('--vid_len', dest='vid_length', 
												type=int,
												default=8, 
												help='Batch size (over multiple gpu)')
		self.parser.add_argument('--mode', dest='mode',
												help='mode to use',
												choices=['xs2xs', 'xss2x', 'edge'],
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
												default=80)
		self.parser.add_argument('--sharp_w', dest='sharp_weight',
												help='training optimizer loss weigh of feat',
												type=float,
												default=20) 
		self.parser.add_argument('--gdl_w', dest='gdl_weight',
												help='training optimizer loss weigh of gdl',
												type=float,
												default=0)
		self.parser.add_argument('--vgg_w', dest='vgg_weight',
												help='training optimizer loss weigh of vgg',
												type=float,
												default=20)
		self.parser.add_argument('--ce_w', dest='ce_weight',
												help='training optimizer loss weigh of vgg',
												type=float,
												default=30)     
		self.parser.add_argument('--ssim_w', dest='ssim_weight',
												help='training optimizer loss weigh of feat',
												type=float,
												default=20)       
		self.parser.add_argument('--interval', dest='interval',
												help='training optimizer loss weigh of feat',
												type=int,
												choices=[1,2,3,4,5],
												default=1)      



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
												default=0, type=int)

		# resume trained model
		self.parser.add_argument('--r', dest='resume',
												help='whether eval after each training ', 
												action='store_true')
		self.parser.add_argument('--checksession', dest='checksession',
												help='checksession to load model',
												default=1, type=int)
		self.parser.add_argument('--checkepoch', dest='checkepoch',
												help='checkepoch to load model',
												default=1, type=int)
		self.parser.add_argument('--checkepoch_range', dest='checkepoch_range',
												help='whether eval after each training ', 
												action='store_true')
		self.parser.add_argument('--checkepoch_low', dest='checkepoch_low',
												help='checkepoch to load model',
												default=1, type=int)    
		self.parser.add_argument('--checkepoch_up', dest='checkepoch_up',
												help='checkepoch to load model',
												default=20, type=int)
		self.parser.add_argument('--checkpoint', dest='checkpoint',
												help='checkpoint to load model',
												default=0, type=int)
		self.parser.add_argument('--load_dir', dest='load_dir',
												help='directory to load models', default="models",
												type=str)

		############### add subparsers ######################
		subparsers = self.parser.add_subparsers(help='sub-command help', dest='runner')

		# only generator mode
		generator_parser = subparsers.add_parser('gen', help='use generator')
		generator_parser.add_argument('--model', dest='model', 
										default='MyFRRN', 
										help='model to use',
										choices=['GridNet', 'MyFRRN'])  
		generator_parser.add_argument('--o', dest='optimizer', 
										help='training optimizer',
										choices =['adamax','adam', 'sgd'], 
										default="adamax")
		generator_parser.add_argument('--learning_rate', dest='learning_rate', 
										help='starting learning rate',
										default=0.001, type=float)

		# gan mode
		gan_parser = subparsers.add_parser('gan', help='use gan')
		gan_parser.add_argument('--model', dest='model', 
										default='GAN', 
										help='model to use',
										choices=['GAN'])  
		gan_parser.add_argument('--netG', dest='netG', 
										default='MyFRRN', 
										help='model to use',
										choices=['GridNet', 'MyFRRN'])  
		gan_parser.add_argument('--netD', dest='netD', 
										default='multi_scale', 
										help='model to use',
										choices=['multi_scale','multi_scale_img', 'multi_scale_img_seg','motion_img','motion_img_seg'])
		gan_parser.add_argument('--numD', dest='num_D', 
										default=3, 
										help='number of discriminator',
										type=int)
		gan_parser.add_argument('--n_layer_D', dest='n_layer_D', 
										default=2, 
										help='number of discriminator layers',
										type=int)
		gan_parser.add_argument('--oG', dest='optG', 
										help='training optimizer',
										choices =['adamax','adam', 'sgd'], 
										default="adamax")
		gan_parser.add_argument('--oD', dest='optD', 
										help='training optimizer',
										choices =['adamax','adam', 'sgd'], 
										default="sgd")
		gan_parser.add_argument('--lrG', dest='lr_G', 
										help='starting learning rate',
										default=0.001, type=float)
		gan_parser.add_argument('--lrD', dest='lr_D', 
										help='starting learning rate',
										default=0.001, type=float)
		gan_parser.add_argument('--adv_w', dest='adv_weight',
										help='training optimizer loss weigh of gdl',
										type=float,
										default=1)
		gan_parser.add_argument('--adv_feat_w', dest='adv_feat_weight',
										help='training optimizer loss weigh of gdl',
										type=float,
										default=1)
		gan_parser.add_argument('--d_w', dest='d_weight',
										help='training optimizer loss weigh of gdl',
										type=float,
										default=10)
		gan_parser.add_argument('--load_G', dest='load_G',
										help='training optimizer loss weigh of gdl',
										action='store_true')
		gan_parser.add_argument('--load_GANG', dest='load_GANG',
										help='training optimizer loss weigh of gdl',
										action='store_true')


		# only generator mode
		vae_parser = subparsers.add_parser('vae', help='use generator')
		vae_parser.add_argument('--model', dest='model', 
										default='VAE', 
										help='model to use',
										choices=['GridNet', 'MyFRRN', 'VAE','VAE_S'])  
		vae_parser.add_argument('--o', dest='optimizer', 
										help='training optimizer',
										choices =['adamax','adam', 'sgd'], 
										default="adamax")
		vae_parser.add_argument('--learning_rate', dest='learning_rate', 
										help='starting learning rate',
										default=0.001, type=float)
		vae_parser.add_argument('--input_size',
										default=(128, 128),
										type=tuple,
										help='input image size')
		vae_parser.add_argument('--latent_dim',
										default=512,
										type=int,
										help='input image size')
		vae_parser.add_argument('--seg_dim',
										default=4,
										type=int,
										help='input image size')
		vae_parser.add_argument('--seg',
										action='store_true',
										help='input image size')
		vae_parser.add_argument('--disparity',
										action='store_true',
										help='input image size')
		self.initialized = True


	def parse(self, save=True):
		if not self.initialized:
			self.initialize()
		self.opt = self.parser.parse_args()
		return self.opt


# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3,4 python main.py  --disp_interval 100 --mode xs2xs --syn_type inter  --load_dir log/MyFRRN_xs2xs_inter_1_05-11-15:07:20 --bs 48 --nw 8  --checksession 1 --s 1  --checkepoch 30 --checkpoint 1611  gan --d_w 10 --netD multi_scale_img_seg --load_G --lrD 0.005