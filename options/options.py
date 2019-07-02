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
												choices=['cityscape', 'ucf101','vimeo'])
		self.parser.add_argument('--split', dest='split', 
												help='whether eval after each training ', 
												default='train',
												choices=['train','val','test','cycgen'])

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
		self.parser.add_argument('--input_h',
										default=128,
										type=int,
										help='input image size')
		self.parser.add_argument('--input_w',
										default=256,
										type=int,
										help='input image size')		
		self.parser.add_argument('--syn_type', dest='syn_type',
												help='synthesize method',
												choices=['inter', 'extra'],
												default='extra') 
		self.parser.add_argument('--vid_len', dest='vid_length', 
												type=int,
												default=1, 
												help='Batch size (over multiple gpu)')
		self.parser.add_argument('--mode', dest='mode',
												help='mode to use',
												choices=['xs2xs', 'xx2x'],
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
												default=0) 
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
												type=float,
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
		self.parser.add_argument('--cyc_prefix', dest='cyc_prefix',
												help='directory to load models', default="log",
												type=str)
		self.parser.add_argument('--imgout_dir', dest='imgout_dir',
												help='directory to load models', default="log",
												type=str)


		self.parser.add_argument('--cycgen_all', dest='cycgen_all',
												help='whether eval after each training ', 
												action='store_true')
		# resume

		self.parser.add_argument('--ef', dest='effec_flow',
												help='whether eval after each training ', 
												action='store_true')
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



		self.parser.add_argument('--drop_seg', dest='drop_seg',
												help='whether eval after each training ', 
												action='store_true')	

		self.parser.add_argument('--high_res', dest='high_res', 
										help='model to use',
										action='store_true')	
		self.parser.add_argument('--re_ref', dest='re_ref', 
										help='model to use',
										action='store_true')	

		############### add subparsers ######################
		subparsers = self.parser.add_subparsers(help='sub-command help', dest='runner')

		# only generator mode
		generator_parser = subparsers.add_parser('gen', help='use generator')
		generator_parser.add_argument('--model', dest='model', 
										default='MyFRRN', 
										help='model to use',
										choices=['GridNet', 'MyFRRN', 'UNet','SepUNet', 'RefineNet', 'B2SNet'])  
		generator_parser.add_argument('--coarse_model', dest='coarse_model', 
										default='MyFRRN', 
										help='model to use',
										choices=['GridNet', 'MyFRRN', 'UNet','SepUNet'])  					  
		generator_parser.add_argument('--o', dest='optimizer', 
										help='training optimizer',
										choices =['adamax','adam', 'sgd'], 
										default="adamax")
		generator_parser.add_argument('--learning_rate', dest='learning_rate', 
										help='starting learning rate',
										default=0.001, type=float)

		generator_parser.add_argument('--n_sc', dest='n_scales', 
										help='starting learning rate',
										default=2, type=int)

		# only generator mode
		refine_parser = subparsers.add_parser('refine', help='use generator')
		refine_parser.add_argument('--model', dest='model', 
										default='RefineNet', 
										help='model to use',
										choices=['RefineNet'])  
		refine_parser.add_argument('--coarse_model', dest='coarse_model', 
										default='MyFRRN', 
										help='model to use',
										choices=['GridNet', 'MyFRRN', 'UNet','SepUNet'])  
		refine_parser.add_argument('--refine_model', dest='refine_model', 
										default='SRN', 
										help='model to use',
										choices=['SRN', 'SRN2', 'SRN3', 'SRN4', 'SRN4Seg', 'SRN4Sharp',  'AttnRefine', 'AttnBaseRefine', 'MSBaseRefine'])	
		refine_parser.add_argument('--high_res_model', dest='high_res_model', 
										default='HResUnet', 
										help='model to use',
										choices=['HResUnet'])			
		refine_parser.add_argument('--re_ref_model', dest='re_ref_model', 
										default='AttnRefineV2', 
										help='model to use',
										choices=['AttnRefineV2'])		
		refine_parser.add_argument('--lock_coarse', dest='lock_coarse', 
										help='model to use',
										action='store_true')		
		refine_parser.add_argument('--pretrained_low', dest='pretrained_low', 
										help='model to use',
										action='store_true')	
		refine_parser.add_argument('--lock_low', dest='lock_low', 
										help='model to use',
										action='store_true')		
		refine_parser.add_argument('--lock_retrain', dest='lock_retrain', 
										help='model to use',
										action='store_true')	
		refine_parser.add_argument('--pretrained_coarse', dest='pretrained_coarse', 
										help='model to use',
										action='store_true')
		refine_parser.add_argument('--o', dest='optimizer', 
										help='training optimizer',
										choices =['adamax','adam', 'sgd'], 
										default="adamax")
		refine_parser.add_argument('--learning_rate', dest='learning_rate', 
										help='starting learning rate',
										default=0.001, type=float)
		refine_parser.add_argument('--n_sc', dest='n_scales', 
										help='starting learning rate',
										default=2, type=int)
		# weight of losses
		refine_parser.add_argument('--r_l1_w', dest='refine_l1_weight',
												help='training optimizer loss weigh of l1',
												type=float,
												default=20)
		refine_parser.add_argument('--r_vgg_w', dest='refine_vgg_weight',
												help='training optimizer loss weigh of vgg',
												type=float,
												default=5)   
		refine_parser.add_argument('--r_ssim_w', dest='refine_ssim_weight',
												help='training optimizer loss weigh of feat',
												type=float,
												default=5) 
		refine_parser.add_argument('--r_gdl_w', dest='refine_gdl_weight',
												help='training optimizer loss weigh of feat',
												type=float,
												default=20)
		# seperate coarse model from refine
		refine_parser.add_argument('--sep_coarse', dest='seperate_coarse', 
										help='model to use',
										action='store_true')
		refine_parser.add_argument('--c_checksession', dest='coarse_checksession',
												help='checksession to load model',
												default=1, type=int)
		refine_parser.add_argument('--c_checkepoch', dest='coarse_checkepoch',
												help='checkepoch to load model',
												default=1, type=int)  
		refine_parser.add_argument('--c_checkpoint', dest='coarse_checkpoint',
												help='checkpoint to load model',
												default=0, type=int)
		refine_parser.add_argument('--c_load_dir', dest='coarse_load_dir',
												help='directory to load models', default="models",
												type=str)
		refine_parser.add_argument('--c_mode', dest='coarse_mode',
												help='mode to use',
												choices=['xs2xs', 'xx2x'],
												default='xs2xs')


		# only generator mode
		refine_gan_parser = subparsers.add_parser('refine_gan', help='use generator')
		refine_gan_parser.add_argument('--model', dest='model', 
										default='RefineGAN', 
										help='model to use',
										choices=['RefineGAN'])  
		refine_gan_parser.add_argument('--coarse_model', dest='coarse_model', 
										default='MyFRRN', 
										help='model to use',
										choices=['GridNet', 'MyFRRN', 'UNet','SepUNet'])  
		refine_gan_parser.add_argument('--refine_model', dest='refine_model', 
										default='SRN', 
										help='model to use',
										choices=['SRN', 'SRN2', 'SRN3', 'SRN4'])		
		refine_gan_parser.add_argument('--lock_coarse', dest='lock_coarse', 
										help='model to use',
										action='store_true')			  
		refine_gan_parser.add_argument('--o', dest='optimizer', 
										help='training optimizer',
										choices =['adamax','adam', 'sgd'], 
										default="adamax")
		refine_gan_parser.add_argument('--learning_rate', dest='learning_rate', 
										help='starting learning rate',
										default=0.001, type=float)
		refine_gan_parser.add_argument('--n_sc', dest='n_scales', 
										help='starting learning rate',
										default=2, type=int)
		# weight of losses
		refine_gan_parser.add_argument('--r_l1_w', dest='refine_l1_weight',
												help='training optimizer loss weigh of l1',
												type=float,
												default=20) 
		refine_gan_parser.add_argument('--r_vgg_w', dest='refine_vgg_weight',
												help='training optimizer loss weigh of vgg',
												type=float,
												default=5)   
		refine_gan_parser.add_argument('--r_ssim_w', dest='refine_ssim_weight',
												help='training optimizer loss weigh of feat',
												type=float,
												default=5)   
		refine_gan_parser.add_argument('--adv_w', dest='refine_adv_weight',
												help='training optimizer loss weigh of feat',
												type=float,
												default=0.02) 
		refine_gan_parser.add_argument('--d_w', dest='refine_d_weight',
												help='training optimizer loss weigh of feat',
												type=float,
												default=1) 
		refine_gan_parser.add_argument('--numD', dest='num_D', 
										default=2, 
										help='number of discriminator',
										type=int)

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