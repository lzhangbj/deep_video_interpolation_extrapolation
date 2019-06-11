ssh -N -f -L localhost:16000:localhost:6000 linz@ckcpu1.cse.ust.hk
ssh -N -f -L localhost:17000:localhost:7000 linz@ckcpu2.cse.ust.hk




CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4,5,6 python main.py  --disp_interval 100 --mode xs2xs --syn_type extra --bs 48 --nw 8 --ce_w 30   --s 0  gan --adv_w 10 --d_w 10 --netD multi_scale_img --lrD 0.001 --n_layer_D 7 --numD 2 --oD sgd

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 python main.py  --disp_interval 100 --mode xs2xs  --syn_type inter --bs 48 --nw 8  --s 14 --val --checkepoch_range --checkepoch_low 15 --checkepoch_up 20 --checksession 14 --load_dir log/GAN_xs2xs_inter_14_05-29-09:55:43  --checkpoint 1611  gan  --netD motion_img --lrD 0.001 --n_layer_D 4 --numD 1


CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,4 python main.py  --disp_interval 100 --mode xs2xs --syn_type extra --bs 32  --nw 8  --s 0 --epochs 30  --interval 1 --vid_len 4 vae



CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0  python main.py  --disp_interval 100 --mode xs2xs --syn_type inter --bs 60  --nw 5  --s 10  --interval 1 --val --checksession 10 --checkepoch_range --checkpoint 1388 --checkepoch_low 20 --checkepoch_up 25 --load_dir log/MyFRRN_xs2xs_inter_10_06-09-19:41:50 gen --model MyFRRN

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3,4,5  python main.py  --disp_interval 100 --mode xs2xs --syn_type inter --bs 48 --nw 8  --s 10 --interval 1 --epochs 30  gen --model SepUNet



CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6  python main.py  --disp_interval 100 --mode xs2xs --syn_type inter --bs 48  --nw 4  --s 11  --interval 1 --val --checksession 11 --checkepoch_range --checkpoint 1266 --checkepoch_low 20 --checkepoch_up 24 --load_dir log/MyFRRN_xs2xs_inter_11_06-10-00:36:22 gen --model MyFRRN