ssh -N -f -L localhost:16000:localhost:6000 linz@ckcpu1.cse.ust.hk
ssh -N -f -L localhost:17000:localhost:7000 linz@ckcpu2.cse.ust.hk
ssh -N -f -L localhost:19000:localhost:9000 linzha@login.leonhard.ethz.ch



coarse frrn xs2xs: 			1,382,814 
coarse frrn xx2x: 			1,329,670 
refine srn : 				7,323,875
refine srn sharp: 	   	   10,811,843
attn refine     : 		  	  693,923

voxel flow		:			3,821,891
cyclic large	:	   	   19,801,345


i5
both_have avg=7.625983193277311   std=3.7713841383562703
only_for  avg=0.4065546218487395  std=0.7919494476850922
only_back avg=0.20436974789915965 std=0.6281473876387734
none_have avg=0.10527731092436975 std=0.37855414728461545

panet i9
both_have avg=8.992773109243698 std=3.8799063470973203
only_for  avg=1.2324649859943977 std=1.4802317658754633
only_back avg=0.7515686274509804 std=1.1581360248316959
none_have avg=0.5688515406162465 std=0.988813664222764


panet i5
both_have avg=10.212941176470588 std=4.036447467291462
only_for  avg=0.6809579831932773 std=1.029192596939918
only_back avg=0.42828571428571427 std=0.8922353680396177
none_have avg=0.21194957983193277 std=0.5826064853617214


panet i9 area 3000
train 
30732
3.2532864766367307
val
5116
3.2804898460165144



trackrcnn i9 area 6000
train 35700 -> 34084
avg track obj cnt = 1.45
val 6000 -> 5635
avg track obj cnt = 1.38

trackrcnn i9 area 0
train 35700 -> 34084
avg track obj cnt = 4.93
val 6000 -> 5635
avg track obj cnt = 4.83



cleaned_panet statistics
car 578843				6.485
motorcycle 11940		0.134
person 309808			3.471	
bus 10560				0.118
traffic light 97374		1.091
bicycle 60870			0.682
train 755				0.008			
truck 22417				0.251

int 5 val
87 	person
105	person
111 bicycle
114 person
117 person and bicycle
129 clutter person
141 bus
171 clutter person
279 person

int 9 val
75  person
180 truck
204 person
228 bicycle
234 person and car

panet i9 bboxes val
12 short range, non-tracking method is better
24 human
30 large appearance change
51 cars
60 bicycle
66 long range cars
99 human
126 failure case
447 humans 
471 humnas
735 cars
786 humans
963 humans
987 cars appearance
1005 humnas
1035 humans and bicycles
1044, 1053 object scaling and appearance change
1056 completely missing objects not good
1062 short range partially occluded cars not good
1116 short range horizontal human, can
1131 humans
1167 long range horizontal cars
1218 bicycle
1239 bicycle
1275 long range car


################### prepare  #########################
pkl data in /data/linz/proj/Dataset/Cityscape/load_files and obj_coords
see data.py and folder.py


cmds

#################### 1. track and gen train ##############################
python main.py  --disp_interval 100 --mode xs2xs --syn_type inter \
--bs 18  --nw 3  --s 24  --split train  --interval 9  --one_hot_seg --input_h 128 --input_w 128 --epochs 30 --kld_w 20 --n_track 10 \
INTER  --model InterGANNet --coarse_model VAEHRNet --train_coarse --vae --gan --track_gen --track_gen_model TrackGenV2 \
--frame_disc --frame_disc_model FrameSNDiscriminator --train_frame_disc \
	--frame_disc_g_w 0.4 --frame_disc_d_w 0.001 --frame_disc_lr 0.0001 \
--video_disc --video_disc_model VideoSNDiscriminator --train_video_disc \
	--video_disc_g_w 0.4 --video_disc_d_w 0.001 --video_disc_lr 0.0001 \
--frame_det_disc --frame_det_disc_model FrameSNDetDiscriminator --train_frame_det_disc \
	--frame_det_disc_g_w 0.4 --frame_det_disc_d_w 0.001 --frame_det_disc_lr 0.0001  \
--video_det_disc --video_det_disc_model VideoLSSNDetDiscriminator --train_video_det_disc \
	--video_det_disc_g_w 0.4 --video_det_disc_d_w 0.01 --video_det_disc_lr 0.0001

################### 2. track and gen val ###################################
python main.py  --disp_interval 100 --mode xs2xs --syn_type inter \
--bs 1  --nw 1  --s ~  --split val  --interval 9  --one_hot_seg --input_h 128 --input_w 128 --n_track 10 \
--load_dir log/~ --checksession ~ --checkepoch_range --checkepoch_low ~ --checkepoch_up ~ --checkpoint ~ \
INTER  --model InterGANNet --load_model InterGANNet --coarse_model VAEHRNet --load_coarse --vae --gan --track_gen --track_gen_model TrackGenV2\
--frame_disc --frame_disc_model FrameSNDiscriminator --load_frame_disc \
--video_disc --video_disc_model VideoSNDiscriminator --load_video_disc \
--frame_det_disc --frame_det_disc_model FrameSNDetDiscriminator --load_frame_det_disc \
--video_det_disc --video_det_disc_model VideoLSSNDetDiscriminator --load_video_det_disc







if not ( (self.epoch<=20 and self.epoch%5==0) or (self.epoch>20 and self.epoch%3==0) ):
			return 

/cluster/scratch/linzha/model/Data/

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3,4  python main.py  --disp_interval 100 --mode xs2xs --syn_type inter --bs 27  --nw 3  --s 3  --split train  --interval 9  --one_hot_seg --input_h 128 --input_w 128 --load_dir log/InterRefineNet_xs2xs_inter_1_07-16-23\:48\:39 --checksession 1 --checkepoch 19 --checkpoint 1487  INTER  --model InterStage3Net --refine --load_model InterRefineNet --coarse_model HRNet --load_coarse --load_refine --refine_model SRNRefine --n_sc 3 --stage3 --stage3_model MSResAttnRefineV2  --train_stage3 --refine_l1_w 20 --refine_gdl_w 20 --refine_vgg_w 20 --refine_ssim_w 20 --stage3_prop

bsub -I -n 5 -R 'rusage[mem=8000, ngpus_excl_p=6]' -R 'select[gpu_mtotal0>=11000]' python main.py  --disp_interval 100 --mode xs2xs  --syn_type inter --bs 4  --nw 1  --s 1  --interval 9  --epochs 50  --split train  --input_h 128 --input_w 128  --one_hot_seg INTER  --model InterGANNet --coarse_model VAEHRNet --vae --gan --train_coarse --frame_disc --train_frame_disc --video_disc --train_video_disc --frame_disc_g_w 4 --video_disc_g_w 4

bsub -W 12:00 -B -N -n 5 -J sess13 -oo sess13.txt -R 'rusage[mem=8000, ngpus_excl_p=6]' -R 'select[gpu_mtotal0>=11000]'

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python main.py --disp_interval 100 --mode xs2xs --syn_type inter --bs 36 --nw 3 --s 6 --split train --interval 9 --one_hot_seg --input_h 128 --input_w 128 --kld_w 20 --epochs 50 INTER --model InterGANNet --coarse_model VAEHRNet --train_coarse --local_disc --frame_disc --frame_disc_model FrameLocalDiscriminator --train_frame_disc --frame_disc_d_w 1 --frame_disc_lr 0.00005 --frame_disc_g_w 10 --video_disc --video_disc_model VideoLocalDiscriminator --train_video_disc --video_disc_d_w 1 --video_disc_lr 0.00005 --video_disc_g_w 10 --vae --gan --seg_disc > exp_out/InterGAN_6_local_seg_disc_train.txt


--r --load_dir log/InterGANNet_xs2xs_inter_10_07-26-20:30:33 --checksession 10 --checkepoch 12 --checkpoint 735 \

######################################## valing ###########################
python main.py  --disp_interval 100 --mode xs2xs --syn_type inter \
--bs 24  --nw 3  --s ~  --split val  --interval 9  --one_hot_seg --input_h 128 --input_w 128 \
--load_dir log/~ --checksession ~ --checkepoch_range --checkepoch_low ~ --checkepoch_up ~ --checkpoint ~ \
INTER  --model InterGANNet --load_model InterGANNet --coarse_model VAEHRNet --load_coarse --vae --gan \
--frame_disc --frame_disc_model FrameSNDiscriminator --load_frame_disc \
--video_disc --video_disc_model VideoSNDiscriminator --load_video_disc \
--frame_det_disc --frame_det_disc_model FrameSNDetDiscriminator --load_frame_det_disc \
--video_det_disc --video_det_disc_model VideoLSSNDetDiscriminator --load_video_det_disc \


####################################### smaller set of disc loss ##############################
python main.py  --disp_interval 100 --mode xs2xs --syn_type inter \
--bs 30  --nw 3  --s 22  --split train  --interval 9  --one_hot_seg --input_h 128 --input_w 128 --epochs 30 --kld_w 20  \
INTER  --model InterGANNet --coarse_model VAEHRNet --train_coarse --vae --gan \
--frame_disc --frame_disc_model FrameSNDiscriminator --train_frame_disc \
	--frame_disc_g_w 0.2 --frame_disc_d_w 0.00004 --frame_disc_lr 0.0001 \
--video_disc --video_disc_model VideoSNDiscriminator --train_video_disc \
	--video_disc_g_w 0.2 --video_disc_d_w 0.00004 --video_disc_lr 0.0001 \
--frame_det_disc --frame_det_disc_model FrameSNDetDiscriminator --train_frame_det_disc \
	--frame_det_disc_g_w 0.1 --frame_det_disc_d_w 0.0001 --frame_det_disc_lr 0.0001  \
--video_det_disc --video_det_disc_model VideoSNDetDiscriminator --train_video_det_disc \
	--video_det_disc_g_w 1 --video_det_disc_d_w 0.008 --video_det_disc_lr 0.0001


##################################### suitable params ###########################
python main.py  --disp_interval 100 --mode xs2xs --syn_type inter \
--bs 18  --nw 3  --s 24  --split train  --interval 9  --one_hot_seg --input_h 128 --input_w 128 --epochs 30 --kld_w 20 --n_track 10 \
INTER  --model InterGANNet --coarse_model VAEHRNet --train_coarse --vae --gan \
--frame_disc --frame_disc_model FrameSNDiscriminator --train_frame_disc \
	--frame_disc_g_w 0.4 --frame_disc_d_w 0.001 --frame_disc_lr 0.0001 \
--video_disc --video_disc_model VideoSNDiscriminator --train_video_disc \
	--video_disc_g_w 0.4 --video_disc_d_w 0.001 --video_disc_lr 0.0001 \
--frame_det_disc --frame_det_disc_model FrameSNDetDiscriminator --train_frame_det_disc \
	--frame_det_disc_g_w 0.4 --frame_det_disc_d_w 0.001 --frame_det_disc_lr 0.0001  \
--video_det_disc --video_det_disc_model VideoLSSNDetDiscriminator --train_video_det_disc \
	--video_det_disc_g_w 0.4 --video_det_disc_d_w 0.01 --video_det_disc_lr 0.0001

sess31, smaller --video_det_disc_d_w 0.03

################################## eth sess13 ################
# set 1 discarded
python main.py  --disp_interval 100 --mode xs2xs --syn_type inter \
--bs 36  --nw 3  --s 13  --split train  --interval 9  --one_hot_seg --input_h 128 --input_w 128 --epochs 50 --kld_w 20  \
INTER  --model InterGANNet --coarse_model VAEHRNet --train_coarse --vae --gan \
--frame_disc --frame_disc_model FrameSNDiscriminator --train_frame_disc \
	--frame_disc_g_w 0.4 --frame_disc_d_w 0.005 --frame_disc_lr 0.0001 \
--video_disc --video_disc_model VideoSNDiscriminator --train_video_disc \
	--video_disc_g_w 0.4 --video_disc_d_w 0.002 --video_disc_lr 0.0001 \
--frame_det_disc --frame_det_disc_model FrameSNDetDiscriminator --train_frame_det_disc \
	--frame_det_disc_g_w 0.4 --frame_det_disc_d_w 0.003 --frame_det_disc_lr 0.0001  \
--video_det_disc --video_det_disc_model VideoSNDetDiscriminator --train_video_det_disc \
	--video_det_disc_g_w 1 --video_det_disc_d_w 0.008 --video_det_disc_lr 0.0001

# set 1 discarded
python main.py  --disp_interval 100 --mode xs2xs --syn_type inter \
--bs 48  --nw 4  --s 13  --split train  --interval 9  --one_hot_seg --input_h 128 --input_w 128 --epochs 30 --kld_w 20 \
INTER  --model InterGANNet --coarse_model VAEHRNet --train_coarse --vae --gan \
--frame_disc --frame_disc_model FrameSNDiscriminator --train_frame_disc \
	--frame_disc_g_w 0.4 --frame_disc_d_w 0.004 --frame_disc_lr 0.0001 \
--video_disc --video_disc_model VideoSNDiscriminator --train_video_disc \
	--video_disc_g_w 0.4 --video_disc_d_w 0.001 --video_disc_lr 0.0001 \
--frame_det_disc --frame_det_disc_model FrameSNDetDiscriminator --train_frame_det_disc \
	--frame_det_disc_g_w 0.4 --frame_det_disc_d_w 0.003 --frame_det_disc_lr 0.0001 \
--video_det_disc --video_det_disc_model VideoSNDetDiscriminator --train_video_det_disc \
	--video_det_disc_g_w 1 --video_det_disc_d_w 0.02 --video_det_disc_lr 0.0001


################################## cpu1 sess14 #############################
--r --load_dir log/InterGANNet_xs2xs_inter_14_07-29-18:49:49 --checksession 14 --checkepoch 18 --checkpoint 817 \


###########################################################################
# sess 13, 14, 15, 16 ... standard settings
--track_obj_loss --track_obj_w 8

# set 1
python main.py  --disp_interval 100 --mode xs2xs --syn_type inter \
--bs 24  --nw 4  --s 17  --split train  --interval 9  --one_hot_seg --input_h 128 --input_w 128 --epochs 30 --kld_w 20 --n_track 4 \
--track_obj_loss --track_obj_w 8 \
INTER  --model InterGANNet --coarse_model VAEHRNet --train_coarse --vae --gan \
--frame_disc --frame_disc_model FrameSNDiscriminator --train_frame_disc \
	--frame_disc_g_w 0.5 --frame_disc_d_w 0.003 --frame_disc_lr 0.0001  \
--video_disc --video_disc_model VideoSNDiscriminator --train_video_disc \
	--video_disc_g_w 0.5 --video_disc_d_w 0.001 --video_disc_lr 0.0001  \
--frame_det_disc --frame_det_disc_model FrameSNDetDiscriminator --train_frame_det_disc \
	--frame_det_disc_g_w 0.5 --frame_det_disc_d_w 0.003 --frame_det_disc_lr 0.0001  \
--video_det_disc --video_det_disc_model VideoVecSNDetDiscriminator --train_video_det_disc \
	--video_det_disc_g_w 1 --video_det_disc_d_w 0.02 --video_det_disc_lr 0.0001  

# set 2
python main.py  --disp_interval 100 --mode xs2xs --syn_type inter \
--bs 16  --nw 4  --s 18  --split train  --interval 5  --one_hot_seg --input_h 128 --input_w 128 --epochs 30 --kld_w 20 --n_track 6 \
--track_obj_loss --track_obj_w 8 \
INTER  --model InterGANNet --coarse_model VAEHRNet --train_coarse --vae --gan \
--frame_disc --frame_disc_model FrameSNDiscriminator --train_frame_disc \
	--frame_disc_g_w 0.3 --frame_disc_d_w 0.002 --frame_disc_lr 0.0001  \
--video_disc --video_disc_model VideoSNDiscriminator --train_video_disc \
	--video_disc_g_w 0.3 --video_disc_d_w 0.0003 --video_disc_lr 0.0001  \
--frame_det_disc --frame_det_disc_model FrameSNDetDiscriminator --train_frame_det_disc \
	--frame_det_disc_g_w 0.3 --frame_det_disc_d_w 0.002 --frame_det_disc_lr 0.0001  \
--video_det_disc --video_det_disc_model VideoVecSNDetDiscriminator --train_video_det_disc \
	--video_det_disc_g_w 1 --video_det_disc_d_w 0.02 --video_det_disc_lr 0.0001  

#########################################################################################

# set 1# discarded
python main.py  --disp_interval 100 --mode xs2xs --syn_type inter \
--bs 36  --nw 4  --s 15  --split train  --interval 9  --one_hot_seg --input_h 128 --input_w 128 --epochs 30 --kld_w 20 --n_track 4 \
INTER  --model InterGANNet --coarse_model VAEHRNet --train_coarse --vae --gan \
--frame_disc --frame_disc_model FrameSNDiscriminator --train_frame_disc \
	--frame_disc_g_w 0.5 --frame_disc_d_w 0.003 --frame_disc_lr 0.0001  \
--video_disc --video_disc_model VideoSNDiscriminator --train_video_disc \
	--video_disc_g_w 0.5 --video_disc_d_w 0.001 --video_disc_lr 0.0001  \
--frame_det_disc --frame_det_disc_model FrameSNDetDiscriminator --train_frame_det_disc \
	--frame_det_disc_g_w 0.5 --frame_det_disc_d_w 0.003 --frame_det_disc_lr 0.0001  \
--video_det_disc --video_det_disc_model VideoVecSNDetDiscriminator --train_video_det_disc \
	--video_det_disc_g_w 1 --video_det_disc_d_w 0.02 --video_det_disc_lr 0.0001  

# set 2 
python main.py  --disp_interval 100 --mode xs2xs --syn_type inter \
--bs 30  --nw 5  --s 11  --split train  --interval 9  --one_hot_seg --input_h 128 --input_w 128 --epochs 30 --kld_w 20  \
INTER  --model InterGANNet --coarse_model VAEHRNet --train_coarse --vae --gan \
--frame_disc --frame_disc_model FrameSNDiscriminator --train_frame_disc \
	--frame_disc_g_w 0.8 --frame_disc_d_w 0.01 --frame_disc_lr 0.0001  \
--video_disc --video_disc_model VideoSNDiscriminator --train_video_disc \
	--video_disc_g_w 0.8 --video_disc_d_w 0.01 --video_disc_lr 0.0001  \
--frame_det_disc --frame_det_disc_model FrameSNDetDiscriminator --train_frame_det_disc \
	--frame_det_disc_g_w 0.4 --frame_det_disc_d_w 0.01 --frame_det_disc_lr 0.0001  \
--video_det_disc --video_det_disc_model VideoSNDetDiscriminator --train_video_det_disc \
	--video_det_disc_g_w 1 --video_det_disc_d_w 0.01 --video_det_disc_lr 0.0001  

# set 3 local discriminator with sn	
python main.py  --disp_interval 100 --mode xs2xs --syn_type inter \
--bs 30  --nw 5  --s 12  --split train  --interval 9  --one_hot_seg --input_h 128 --input_w 128 --epochs 50 --kld_w 20  \
INTER  --model InterGANNet --coarse_model VAEHRNet --train_coarse --vae --gan \
--frame_disc --frame_disc_model FrameSNLocalDiscriminator --train_frame_disc \
	--frame_disc_g_w 0.8 --frame_disc_d_w 0.001 --frame_disc_lr 0.0001  \
--video_disc --video_disc_model VideoSNLocalDiscriminator --train_video_disc \
	--video_disc_g_w 0.8 --video_disc_d_w 0.0001 --video_disc_lr 0.0001  


--load_coarse --load_frame_disc --load_frame_det_disc --load_video_disc --load_video_det_disc --load_model InterGANNet

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2  python main.py  --disp_interval 100 --mode xs2xs --syn_type inter --bs 30  --nw 3  --s 10  --split train  --interval 9  --one_hot_seg --input_h 128 --input_w 128 --epochs 50 --kld_w 20  INTER  --model InterGANNet --coarse_model VAEHRNet --train_coarse --vae --gan --frame_disc --frame_disc_model FrameSNDiscriminator --train_frame_disc --frame_disc_g_w 0.8 --frame_disc_d_w 0.1 --frame_disc_lr 0.0001  --video_disc --video_disc_model VideoSNDiscriminator --train_video_disc --video_disc_g_w 0.8 --video_disc_d_w 0.1 --video_disc_lr 0.0001  --frame_det_disc --frame_det_disc_model FrameSNDetDiscriminator --train_frame_det_disc --frame_det_disc_g_w 0.4 --frame_det_disc_d_w 0.1 --frame_det_disc_lr 0.0001  --video_det_disc --video_det_disc_model VideoSNDetDiscriminator --train_video_det_disc --video_det_disc_g_w 0.8 --video_det_disc_d_w 0.1 --video_det_disc_lr 0.0001