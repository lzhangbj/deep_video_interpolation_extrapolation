ssh -N -f -L localhost:16000:localhost:6000 linz@ckcpu1.cse.ust.hk
ssh -N -f -L localhost:17000:localhost:7000 linz@ckcpu2.cse.ust.hk




CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4,5,6 python main.py  --disp_interval 100 --mode xs2xs --syn_type extra --bs 48 --nw 8 --ce_w 30   --s 0  gan --adv_w 10 --d_w 10 --netD multi_scale_img --lrD 0.001 --n_layer_D 7 --numD 2 --oD sgd

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 python main.py  --disp_interval 100 --mode xs2xs  --syn_type inter --bs 48 --nw 8  --s 14 --val --checkepoch_range --checkepoch_low 15 --checkepoch_up 20 --checksession 14 --load_dir log/GAN_xs2xs_inter_14_05-29-09:55:43  --checkpoint 1611  gan  --netD motion_img --lrD 0.001 --n_layer_D 4 --numD 1


CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,4 python main.py  --disp_interval 100 --mode xs2xs --syn_type extra --bs 32  --nw 8  --s 0 --epochs 30  --interval 1 --vid_len 4 vae