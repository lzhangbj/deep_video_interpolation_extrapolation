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
												choices=['train','val','test','cycgen', 'mycycgen'])
		self.parser.add_argument('--img_dir', dest='img_dir',
												help='directory to load models', default=None,
												type=str)
		self.parser.add_argument('--seg_dir', dest='seg_dir',
												help='directory to load models', default=None,
												type=str)
		self.parser.add_argument('--cycgen_load_dir', dest='cycgen_load_dir',
												help='directory to load cycgen inputs', default=None,
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
		self.parser.add_argument('--one_hot_seg', dest='one_hot_seg',
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
												help='whether resume ', 
												action='store_true')
		self.parser.add_argument('--checksession', dest='checksession',
												help='checksession to load model',
												default=1, type=int)
		self.parser.add_argument('--checkepoch', dest='checkepoch',
												help='checkepoch to load model',
												default=1, type=int)
		self.parser.add_argument('--checkepoch_range', dest='checkepoch_range',
												help='whether eval multiple epochs', 
												action='store_true')
		self.parser.add_argument('--checkepoch_low', dest='checkepoch_low',
												help='when checkepoch_range is true, inclusive starting epoch',
												default=1, type=int)    
		self.parser.add_argument('--checkepoch_up', dest='checkepoch_up',
												help='when checkepoch_range is true, inclusive ending epoch',
												default=20, type=int)
		self.parser.add_argument('--checkpoint', dest='checkpoint',
												help='checkpoint to load model',
												default=0, type=int)
		self.parser.add_argument('--load_dir', dest='load_dir',
												help='directory to load models', default="models",
												type=str)
		# weight of losses
		self.parser.add_argument('--l1_w', dest='l1_weight',
												help='training optimizer loss weigh of l1',
												type=float,
												default=80)
		self.parser.add_argument('--gdl_w', dest='gdl_weight',
												help='training optimizer loss weigh of gdl',
												type=float,
												default=80)
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
		self.parser.add_argument('--kld_w', dest='kld_weight',
												help='training optimizer loss weigh of feat',
												type=float,
												default=20) 
		self.parser.add_argument('--track_obj_loss', dest='track_obj_loss',
										help='whether load coarse model ', 
										action='store_true')      
		self.parser.add_argument('--track_obj_w', dest='track_obj_weight',
												help='training optimizer loss weigh of feat',
												type=float,
												default=80) 

		self.parser.add_argument('--vid_len', dest='vid_length', 
												type=int,
												default=1, 
												help='predicted video length')
		self.parser.add_argument('--n_track', dest='num_track_per_img', 
												type=int,
												default=4, 
												help='predicted video length')

		self.parser.add_argument('--highres_large', dest='highres_large',
										help='whether load coarse model ', 
										action='store_true')

		############### add subparsers ######################
		subparsers = self.parser.add_subparsers(help='sub-command help', dest='runner')

		##############################################################################
		############################# extrapolation subparser ########################
		##############################################################################
		extra_parser = subparsers.add_parser('EXTRA', help='use extrapolation')
		extra_parser.add_argument('--model', dest='model', 
										default='ExtraNet', 
										help='model to use',
										choices=['ExtraNet', 'ExtraInpaintNet'])  
		extra_parser.add_argument('--load_model', dest='load_model', 
										default='ExtraNet', 
										help='model to use',
										choices=['ExtraNet', 'ExtraInpaintNet'])  

		### coarse model settings ###
		extra_parser.add_argument('--coarse_model', dest='coarse_model', 
										default='HRNet', 
										help='model to use',
										choices=['HRNet'])  					  
		extra_parser.add_argument('--coarse_o', dest='coarse_optimizer', 
										help='training coarse optimizer',
										choices =['adamax','adam', 'sgd'], 
										default="adamax")
		extra_parser.add_argument('--coarse_lr', dest='coarse_learning_rate', 
										help='coarse learning rate',
										default=0.001, type=float)	
		extra_parser.add_argument('--load_coarse', dest='load_coarse',
												help='whether load coarse model ', 
												action='store_true')
		extra_parser.add_argument('--train_coarse', dest='train_coarse',
												help='whether train coarse model ', 
												action='store_true')

		### inpaint model settings ###
		extra_parser.add_argument('--inpaint', dest='inpaint', 
										help='whether inpaint or not',
										action='store_true')
		extra_parser.add_argument('--inpaint_mask', dest='inpaint_mask', 
										help='whether inpaint mask or not',
										action='store_true')	  
		extra_parser.add_argument('--inpaint_model', dest='inpaint_model', 
										default='InpaintUnet', 
										help='inpaint model to use',
										choices=['InpaintUnet'])  					  
		extra_parser.add_argument('--inpaint_o', dest='inpaint_optimizer', 
										help='training inpaint optimizer',
										choices =['adamax','adam', 'sgd'], 
										default="adamax")
		extra_parser.add_argument('--inpaint_lr', dest='inpaint_learning_rate', 
										help='inpaint learning rate',
										default=0.001, type=float)	
		extra_parser.add_argument('--load_inpaint', dest='load_inpaint',
												help='whether load inpaint model ', 
												action='store_true')	
		extra_parser.add_argument('--train_inpaint', dest='train_inpaint',
												help='whether train inpaint model ', 
												action='store_true')	
		extra_parser.add_argument('--num_pred_once', dest='num_pred_once',
												help='#frames extrapolated for each model output', default=1,
												type=int)
		extra_parser.add_argument('--num_pred_step', dest='num_pred_step',
												help='#steps extrapolated', default=1,
												type=int)
		extra_parser.add_argument('--fix_init_frames', dest='fix_init_frames',
												help='wheter add start frames always', 
												action='store_true')


		##############################################################################
		############################ interpolation subparser #########################
		##############################################################################
		inter_parser = subparsers.add_parser('INTER', help='use extrapolation')
		inter_parser.add_argument('--model', dest='model', 
										default='InterNet', 
										help='model to use',
										choices=['InterNet', 'InterRefineNet', 'InterStage3Net', 'InterGANNet'])  
		inter_parser.add_argument('--load_model', dest='load_model', 
										default='InterNet', 
										help='model to use',
										choices=['InterNet', 'InterRefineNet', 'InterStage3Net', 'InterGANNet'])  
		inter_parser.add_argument('--n_sc', dest='n_scales', 
										help='scales of output',
										default=1, type=int)	


		inter_parser.add_argument('--gan', dest='gan',
												help='whether load coarse model ', 
												action='store_true')

		### coarse model settings ###
		inter_parser.add_argument('--coarse_model', dest='coarse_model', 
										default='HRNet', 
										help='model to use',
										choices=['HRNet', 'VAEHRNet'])  					  
		inter_parser.add_argument('--coarse_o', dest='coarse_optimizer', 
										help='training coarse optimizer',
										choices =['adamax','adam', 'sgd'], 
										default="adamax")
		inter_parser.add_argument('--coarse_lr', dest='coarse_learning_rate', 
										help='coarse learning rate',
										default=0.001, type=float)	
		inter_parser.add_argument('--load_coarse', dest='load_coarse',
												help='whether load coarse model ', 
												action='store_true')
		inter_parser.add_argument('--train_coarse', dest='train_coarse',
												help='whether train coarse model ', 
												action='store_true')
		inter_parser.add_argument('--vae', dest='vae',
												help='whether train coarse model ', 
												action='store_true')

		inter_parser.add_argument('--seg_disc', dest='seg_disc',
												help='whether train coarse model ', 
												action='store_true')

		inter_parser.add_argument('--track_gen', dest='track_gen',
												help='whether train coarse model ', 
												action='store_true')
		inter_parser.add_argument('--track_gen_model', dest='track_gen_model', 
										default='TrackGen', 
										help='model to use',
										choices=['TrackGen','TrackGenV2'])  
		inter_parser.add_argument('--loc_diff_w', dest='loc_diff_weight', 
										help='coarse learning rate',
										default=100, type=float)


		### refine model settings ###
		inter_parser.add_argument('--refine', dest='refine', 
										help='whether refine or not',
										action='store_true')  
		inter_parser.add_argument('--with_gt_seg', dest='with_gt_seg', 
										help='whether refine',
										action='store_true')
		inter_parser.add_argument('--refine_model', dest='refine_model', 
										default='refineUnet', 
										help='refine model to use',
										choices=['refineUnet', 'SRNRefine'])  					  
		inter_parser.add_argument('--refine_o', dest='refine_optimizer', 
										help='training refine optimizer',
										choices =['adamax','adam', 'sgd'], 
										default="adamax")
		inter_parser.add_argument('--refine_lr', dest='refine_learning_rate', 
										help='refine learning rate',
										default=0.001, type=float)	
		inter_parser.add_argument('--load_refine', dest='load_refine',
												help='whether load refine model ', 
												action='store_true')	
		inter_parser.add_argument('--train_refine', dest='train_refine',
												help='whether train refine model ', 
												action='store_true')
		# weight of losses
		inter_parser.add_argument('--refine_l1_w', dest='refine_l1_weight',
												help='training optimizer loss weigh of l1',
												type=float,
												default=80)
		inter_parser.add_argument('--refine_gdl_w', dest='refine_gdl_weight',
												help='training optimizer loss weigh of gdl',
												type=float,
												default=80)
		inter_parser.add_argument('--refine_vgg_w', dest='refine_vgg_weight',
												help='training optimizer loss weigh of vgg',
												type=float,
												default=20)   
		inter_parser.add_argument('--refine_ssim_w', dest='refine_ssim_weight',
												help='training optimizer loss weigh of feat',
												type=float,
												default=20) 

		### stage3 model ###
		inter_parser.add_argument('--stage3', dest='stage3', 
										help='whether refine or not',
										action='store_true')  
		inter_parser.add_argument('--train_stage3', dest='train_stage3', 
										help='whether refine or not',
										action='store_true') 
		inter_parser.add_argument('--load_stage3', dest='load_stage3', 
										help='whether refine or not',
										action='store_true')  
		inter_parser.add_argument('--stage3_model', dest='stage3_model', 
										default='MSResAttnRefine', 
										help='refine model to use',
										choices=['MSResAttnRefine', 'MSResAttnRefineV2','MSResAttnRefineV2Base','MSResAttnRefineV3'])
		inter_parser.add_argument('--stage3_prop', dest='stage3_prop', 
										help='whether refine or not',
										action='store_true') 
		inter_parser.add_argument('--stage3_flow_consist_w', dest='stage3_flow_consist_weight', 
										help='whether refine or not',
										type=float,
										default=0)


		inter_parser.add_argument('--local_disc', dest='local_disc', 
										help='whether refine or not',
										action='store_true') 

		### frame_disc model ###
		inter_parser.add_argument('--frame_disc', dest='frame_disc', 
										help='whether refine or not',
										action='store_true')  
		inter_parser.add_argument('--frame_disc_o', dest='frame_disc_optimizer', 
										help='training refine optimizer',
										choices =['adamax','adam', 'sgd'], 
										default="adamax")
		inter_parser.add_argument('--frame_disc_lr', dest='frame_disc_learning_rate', 
										help='refine learning rate',
										default=0.001, type=float)
		inter_parser.add_argument('--train_frame_disc', dest='train_frame_disc', 
										help='whether refine or not',
										action='store_true') 
		inter_parser.add_argument('--load_frame_disc', dest='load_frame_disc', 
										help='whether refine or not',
										action='store_true')  
		inter_parser.add_argument('--load_frame_disc_model', dest='load_frame_disc_model', 
										default='FrameDiscriminator', 
										help='refine model to use',
										choices=['FrameDiscriminator', 'FrameLocalDiscriminator', 
												'FrameSNDiscriminator', 'FrameSNLocalDiscriminator', 
												'FrameDetDiscriminator', 'FrameSNDetDiscriminator'])  
		inter_parser.add_argument('--frame_disc_model', dest='frame_disc_model', 
										default='FrameDiscriminator', 
										help='refine model to use',
										choices=['FrameDiscriminator', 'FrameLocalDiscriminator', 
												'FrameSNDiscriminator', 'FrameSNLocalDiscriminator', 
												'FrameDetDiscriminator', 'FrameSNDetDiscriminator'])
		inter_parser.add_argument('--frame_disc_d_w', dest='frame_disc_disc_weight', 
										help='whether refine or not',
										type=float,
										default=1) 
		inter_parser.add_argument('--frame_disc_g_w', dest='frame_disc_gen_weight', 
										help='whether refine or not',
										type=float,
										default=1)
		### frame_det_disc model ###
		inter_parser.add_argument('--frame_det_disc', dest='frame_det_disc', 
										help='whether refine or not',
										action='store_true')  
		inter_parser.add_argument('--frame_det_disc_o', dest='frame_det_disc_optimizer', 
										help='training refine optimizer',
										choices =['adamax','adam', 'sgd'], 
										default="adamax")
		inter_parser.add_argument('--frame_det_disc_lr', dest='frame_det_disc_learning_rate', 
										help='refine learning rate',
										default=0.001, type=float)
		inter_parser.add_argument('--train_frame_det_disc', dest='train_frame_det_disc', 
										help='whether refine or not',
										action='store_true') 
		inter_parser.add_argument('--load_frame_det_disc', dest='load_frame_det_disc', 
										help='whether refine or not',
										action='store_true')  
		inter_parser.add_argument('--load_frame_det_disc_model', dest='load_frame_det_disc_model', 
										default='FrameDiscriminator', 
										help='refine model to use',
										choices=['FrameDiscriminator', 'FrameLocalDiscriminator', 
												'FrameSNDiscriminator', 'FrameSNLocalDiscriminator', 
												'FrameDetDiscriminator', 'FrameSNDetDiscriminator','FrameLSSNDetDiscriminator'])  
		inter_parser.add_argument('--frame_det_disc_model', dest='frame_det_disc_model', 
										default='FrameDiscriminator', 
										help='refine model to use',
										choices=['FrameDiscriminator', 'FrameLocalDiscriminator', 
												'FrameSNDiscriminator', 'FrameSNLocalDiscriminator', 
												'FrameDetDiscriminator', 'FrameSNDetDiscriminator','FrameLSSNDetDiscriminator'])
		inter_parser.add_argument('--frame_det_disc_d_w', dest='frame_det_disc_disc_weight', 
										help='whether refine or not',
										type=float,
										default=1) 
		inter_parser.add_argument('--frame_det_disc_g_w', dest='frame_det_disc_gen_weight', 
										help='whether refine or not',
										type=float,
										default=1) 


		### video_disc model ###
		inter_parser.add_argument('--video_disc', dest='video_disc', 
										help='whether refine or not',
										action='store_true')  
		inter_parser.add_argument('--video_disc_o', dest='video_disc_optimizer', 
										help='training refine optimizer',
										choices =['adamax','adam', 'sgd'], 
										default="adamax")
		inter_parser.add_argument('--video_disc_lr', dest='video_disc_learning_rate', 
										help='refine learning rate',
										default=0.001, type=float)
		inter_parser.add_argument('--train_video_disc', dest='train_video_disc', 
										help='whether refine or not',
										action='store_true') 
		inter_parser.add_argument('--load_video_disc', dest='load_video_disc', 
										help='whether refine or not',
										action='store_true')   
		inter_parser.add_argument('--load_video_disc_model', dest='load_video_disc_model', 
										default='VideoDiscriminator', 
										help='refine model to use',
										choices=['VideoDiscriminator', 'VideoLocalDiscriminator', 
												'VideoSNDiscriminator', 'VideoSNLocalDiscriminator', 
												'VideoDetDiscriminator','VideoSNDetDiscriminator',
												'VideoLSSNDetDiscriminator','VideoVecSNDetDiscriminator','VideoPoolSNDetDiscriminator'])  
		inter_parser.add_argument('--video_disc_model', dest='video_disc_model', 
										default='VideoDiscriminator', 
										help='refine model to use',
										choices=['VideoDiscriminator','VideoLocalDiscriminator', 
												'VideoSNDiscriminator', 'VideoSNLocalDiscriminator',
												'VideoDetDiscriminator','VideoSNDetDiscriminator', 
												'VideoLSSNDetDiscriminator','VideoVecSNDetDiscriminator','VideoPoolSNDetDiscriminator'])
		inter_parser.add_argument('--video_disc_d_w', dest='video_disc_disc_weight', 
										help='whether refine or not',
										type=float,
										default=1) 
		inter_parser.add_argument('--video_disc_g_w', dest='video_disc_gen_weight', 
										help='whether refine or not',
										type=float,
										default=1) 

		### video_det_disc model ###
		inter_parser.add_argument('--video_det_disc', dest='video_det_disc', 
										help='whether refine or not',
										action='store_true')  
		inter_parser.add_argument('--video_det_disc_o', dest='video_det_disc_optimizer', 
										help='training refine optimizer',
										choices =['adamax','adam', 'sgd'], 
										default="adamax")
		inter_parser.add_argument('--video_det_disc_lr', dest='video_det_disc_learning_rate', 
										help='refine learning rate',
										default=0.001, type=float)
		inter_parser.add_argument('--train_video_det_disc', dest='train_video_det_disc', 
										help='whether refine or not',
										action='store_true') 
		inter_parser.add_argument('--load_video_det_disc', dest='load_video_det_disc', 
										help='whether refine or not',
										action='store_true')   
		inter_parser.add_argument('--load_video_det_disc_model', dest='load_video_det_disc_model', 
										default='VideoDiscriminator', 
										help='refine model to use',
										choices=['VideoDiscriminator', 'VideoLocalDiscriminator', 
												'VideoSNDiscriminator', 'VideoSNLocalDiscriminator', 
												'VideoDetDiscriminator','VideoSNDetDiscriminator',
												'VideoLSSNDetDiscriminator', 'VideoLocalPatchSNDetDiscriminator',
												'VideoVecSNDetDiscriminator', 'VideoPoolSNDetDiscriminator',
												'VideoGlobalZeroSNDetDiscriminator', 'VideoGlobalResSNDetDiscriminator',
												'VideoGlobalMaskSNDetDiscriminator', 'VideoGlobalCoordSNDetDiscriminator'])  
		inter_parser.add_argument('--video_det_disc_model', dest='video_det_disc_model', 
										default='VideoDiscriminator', 
										help='refine model to use',
										choices=['VideoDiscriminator','VideoLocalDiscriminator', 
												'VideoSNDiscriminator', 'VideoSNLocalDiscriminator',
												'VideoDetDiscriminator','VideoSNDetDiscriminator', 
												'VideoLSSNDetDiscriminator', 'VideoLocalPatchSNDetDiscriminator',
												'VideoVecSNDetDiscriminator', 'VideoPoolSNDetDiscriminator',
												'VideoGlobalZeroSNDetDiscriminator', 'VideoGlobalResSNDetDiscriminator',
												'VideoGlobalMaskSNDetDiscriminator', 'VideoGlobalCoordSNDetDiscriminator'])
		inter_parser.add_argument('--video_det_disc_d_w', dest='video_det_disc_disc_weight', 
										help='whether refine or not',
										type=float,
										default=1) 
		inter_parser.add_argument('--video_det_disc_g_w', dest='video_det_disc_gen_weight', 
										help='whether refine or not',
										type=float,
										default=1) 



		self.initialized = True


	def parse(self, save=True):
		if not self.initialized:
			self.initialize()
		self.opt = self.parser.parse_args()
		return self.opt


# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3,4 python main.py  --disp_interval 100 --mode xs2xs --syn_type inter  --load_dir log/MyFRRN_xs2xs_inter_1_05-11-15:07:20 --bs 48 --nw 8  --checksession 1 --s 1  --checkepoch 30 --checkpoint 1611  gan --d_w 10 --netD multi_scale_img_seg --load_G --lrD 0.005