import pickle
import json
import shutil
import numpy as np
import numpy.random as nr
from PIL import Image
import sys
import os
import collections
import cv2
from os.path import isfile, join
from utils.data_utils import color_map
import glob

def vis_seg_mask(seg, n_classes=20):
	'''
		mask (bs, h,w) into normed rgb (bs, h,w,3)
		all tensors
	'''
	global color_map
	assert len(seg.shape) == 3
	seg = seg.astype(np.uint32)
	# print(seg.dtype)
	rgb_seg = np.array(color_map)[seg]
	return rgb_seg.astype(np.uint8)

def record_eff_img(store_dir):
	types = ['train', 'val']
	store_dict = {'train':[],
					'val':[]    }
	for type in types:
		names = glob.glob("/data/agong/{}/large_flow/*/*.npy".format(type), recursive=True)
		# print("{} {}".format(type, len(names)))
		for name in names:
			name_splits = name.split('/')
			name_splits[-1] =name_splits[-1][:-9]
			name = type+'/'+name_splits[-2] + '/' + name_splits[-1]
			store_dict[type].append(name)
	with open(store_dir, 'wb') as f:
		pickle.dump(store_dict, f)

# record_eff_img('large_optical_flow.pkl') 

def check_record(dir):
	with open(dir, 'rb') as f:
		file=pickle.load(f)

	print("train\t{} {:.4f}".format(len(file['train']), len(file['train'])/(2975*28)))
	print("val\t{} {:.4f}".format(len(file['val']), len(file['val'])/(500*28)))

# check_record('/data/linz/proj/Dataset/Cityscape/optical_flow/effec_optical_flow.pkl')


def filter_load_file(load_file_dir, flow_file_dir, effec_load_file_dir):
	with open(load_file_dir, 'rb') as f:
		load_file = pickle.load(f)

	with open(flow_file_dir, 'rb') as f:
		flow_file = pickle.load(f)
	
	effec_load_file ={'train':[],
						'val':[]}
	types = ['train','val']
	for type in types:
		for seq in load_file[type]:
			valid = True
			if seq[0] not in flow_file[type]:
				valid = False
			if valid:
				effec_load_file[type].append(seq)
		print("{} {} --> {} {:.4f}".format(type, len(load_file[type]), len(effec_load_file[type]), \
							len(effec_load_file[type])/float(len(load_file[type]))))

	with open(effec_load_file_dir, 'wb') as f:
		pickle.dump(effec_load_file, f)

# filter_load_file('/data/linz/proj/Dataset/Cityscape/load_files/int_1_len_3_extra_lsclip.pkl',
#               '/data/linz/proj/Dataset/Cityscape/optical_flow/effec_optical_flow.pkl',
#               '/data/linz/proj/Dataset/Cityscape/load_files/effec_flow_int_1_len_3_extra_lsclip.pkl')


def test_flow(dir):
	files = glob.glob(dir+'/*/*.npy')
	for file in files:
		flow_image = np.load(file)
		print(flow_image.shape)
		print(flow_image.dtype)
		print(np.max(flow_image))
		print(np.min(flow_image))
		break

# test_flow('/data/agong/train/flow_npz')


def rec_region(dir):
	names = glob.glob("{}/*/*.npy".format(dir), recursive=True)
	print('ori train images {}'.format(len(names))) 
	for file in names:
		flow_img = np.load(file)
		flow_abs_img = np.linalg.norm(flow_img, axis=0)
		print(np.mean(flow_abs_img))
		# invalid_mask = (flow_abs_img > 100)
		# invalid_h, invalid_w =  np.nonzero(invalid_mask)
		# h_min = np.min(invalid_h)
		# h_max = np.max(invalid_h)
		# w_min = np.min(invalid_w)
		# w_max = np.max(invalid_w)

		# print(h_min, w_min, h_max, w_max)

# rec_region('/data/agong/train/large_flow')

def imgs2vid(pathIn, fps, res, interval=2):
	pathOut = pathIn
	img_files = glob.glob(pathIn + "*.png")
	start_ind = int(pathIn[-6:])

	i = 1
	name = pathIn[:-6] + "{:0>6d}_*.png".format(start_ind+i)
	file_name_list = glob.glob(name)
	file_name_list.sort()
	while len(file_name_list) > 0:
		img_files = img_files + file_name_list
		i+=1
		name = pathIn[:-6] + "{:0>6d}*.png".format(start_ind+i)
		file_name_list = glob.glob(name)
		file_name_list.sort()

	# gt
	gt_files = img_files.copy()
	for file in gt_files:
		if 'pred' in file:
			gt_files.remove(file)

	if interval > 1:
		assert len(gt_files) == 29, len(gt_files)
		frame_array = []
		for i in range(len(gt_files)):
			filename=gt_files[i]
			#reading each files
			img = cv2.imread(filename)
			height, width, layers = img.shape
			size = (width,height)
			
			#inserting the frames into an image array
			frame_array.append(img)
		out = cv2.VideoWriter(pathOut+"_{}_gt.avi".format(res),cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
		for i in range(len(frame_array)):
			# writing to a image array
			out.write(frame_array[i])
		out.release()


	# pred
	pred_files = img_files.copy()
	for file in pred_files:
		if 'gt' in file:
			pred_files.remove(file)
	if interval > 1:
		assert len(pred_files) == 29, len(pred_files)
		frame_array = []
		for i in range(len(pred_files)):
			filename=pred_files[i]
			#reading each files
			img = cv2.imread(filename)
			height, width, layers = img.shape
			size = (width,height)
			
			#inserting the frames into an image array
			frame_array.append(img)
		out = cv2.VideoWriter(pathOut+"_{}_pred.avi".format(res),cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
		for i in range(len(frame_array)):
			# writing to a image array
			out.write(frame_array[i])
		out.release()

	# gt and pred
	interm = int(1/interval)
	if interm >= 1:
		for loc, file in enumerate(gt_files[:-1]):
			for i in range(interm):
				gt_files.insert((interm+1)*loc + 1, file)

	assert len(gt_files) == len(pred_files), [ len(gt_files), len(pred_files) ]

	frame_array = []
	for i in range(len(pred_files)):
		gt_filename=gt_files[i]
		pred_filename=pred_files[i]
		#reading each files
		gt_img = cv2.imread(gt_filename)
		pred_img = cv2.imread(pred_filename)
		height, width, layers = gt_img.shape
		comb_img = np.concatenate([gt_img, pred_img], axis=0)
		size = (width,height*2)
		
		#inserting the frames into an image array
		frame_array.append(comb_img)
	out = cv2.VideoWriter(pathOut+"_{}_comb.avi".format(res),cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
	for i in range(len(frame_array)):
		# writing to a image array
		out.write(frame_array[i])
	out.release()


# clip_file = "/data/linz/proj/Dataset/Cityscape/load_files/root_clip.pkl"
# with open(clip_file, 'rb') as f:
#   clips = pickle.load(f)
#   clips = clips['val']
# res='256x512'
# for step, clip in enumerate(clips):
#   if step > 60:
#       break
#   imgs2vid("cycgen/cityscape/{}/{}".format(res, clip[0]), 17*2, res, interval=2)
#   sys.stdout.write("\r {} / 60".format(step))


def make_vid_dirs(res):
	res_str = "{}x{}".format(res[1], res[0])
	clip_file = "/data/linz/proj/Dataset/Cityscape/load_files/root_clip.pkl"
	with open(clip_file, 'rb') as f:
		clips = pickle.load(f)
		clips = clips['val']
	prefix = '/data/linz/proj/Dataset/Cityscape/leftImg_sequence/leftImg8bit_sequence'
	save_dir = 'cycgen/cityscape/{:d}x{:d}/gt'.format(res[1], res[0])
	for clip_cnt, clip in enumerate(clips):
		if clip_cnt > 60:
			break
		# make clip dir
		start_img_dir = clip[0]
		start_img_dir = start_img_dir.split('/')
		start_img_dir[-1] = '_'.join(start_img_dir[-1].split('_')[:3])
		clip_dir = '/'.join(start_img_dir)
		save_clip_dir = os.path.join(save_dir, clip_dir)
		if not os.path.exists(save_clip_dir):
			os.makedirs(save_clip_dir)
		# load images
		for cnt, img_name in enumerate(clip):
			img_dir = os.path.join(prefix, img_name+"_leftImg8bit.png")
			with open(img_dir, 'rb') as f:
				img = Image.open(f)
				img = img.convert('RGB')
			img_resize = img.resize(res, Image.BILINEAR)
			save_img_name = os.path.join(save_clip_dir, "{:0>2d}.0.png".format(cnt))
			img_resize.save(save_img_name)
		sys.stdout.write('\r saving {} {}'.format(clip_cnt, save_img_name))

# make_vid_dirs((256, 128))

def make_vid_seg_dirs(res):
	res_str = "{}x{}".format(res[1], res[0])
	clip_file = "/data/linz/proj/Dataset/Cityscape/load_files/root_clip.pkl"
	with open(clip_file, 'rb') as f:
		clips = pickle.load(f)
		clips = clips['val']
	prefix = '/data/linz/proj/Dataset/Cityscape/leftImg_sequence/gtFine_myseg_id_sequence_128x256'
	save_dir = 'cycgen/cityscape/{:d}x{:d}/gt_seg'.format(res[1], res[0])
	for clip_cnt, clip in enumerate(clips):
		if clip_cnt > 60:
			break
		# make clip dir
		start_img_dir = clip[0]
		start_img_dir = start_img_dir.split('/')
		start_img_dir[-1] = '_'.join(start_img_dir[-1].split('_')[:3])
		clip_dir = '/'.join(start_img_dir)
		save_clip_dir = os.path.join(save_dir, clip_dir)
		if not os.path.exists(save_clip_dir):
			os.makedirs(save_clip_dir)
		# load images
		for cnt, img_name in enumerate(clip):
			img_dir = os.path.join(prefix, img_name+"_gtFine_myseg_id.png")
			with open(img_dir, 'rb') as f:
				img = Image.open(f)
				img = img.convert('L')
			# img_resize = img.resize(res, Image.NEAREST)
			save_img_name = os.path.join(save_clip_dir, "{:0>2d}.0.png".format(cnt))
			img.save(save_img_name)
		sys.stdout.write('\r saving {} {}'.format(clip_cnt, save_img_name))

# make_vid_seg_dirs((256, 128))

def resize_imgs(res):
	res_str = "{}x{}".format(res[1], res[0])
	clip_file = "/data/linz/proj/Dataset/Cityscape/load_files/root_clip.pkl"
	with open(clip_file, 'rb') as f:
		clips = pickle.load(f)
	clips = clips['val']
	# prefix = '/data/linz/proj/Dataset/Cityscape/leftImg_sequence/leftImg8bit_sequence'
	prefix = '/data/linz/proj/Dataset/Cityscape/leftImg_sequence/gtFine_myseg_id_sequence'
	# save_dir = '/data/linz/proj/Dataset/Cityscape/leftImg_sequence/leftImg8bit_sequence_{:d}x{:d}'.format(res[1], res[0])
	save_dir = '/data/linz/proj/Dataset/Cityscape/leftImg_sequence/gtFine_myseg_id_sequence_{:d}x{:d}'.format(res[1], res[0])
	for clip_cnt, clip in enumerate(clips):
		# make clip dir
		start_img_dir = clip[0]
		clip_dir = '/'.join(start_img_dir.split('/')[:-1])
		save_clip_dir = os.path.join(save_dir, clip_dir)
		if not os.path.exists(save_clip_dir):
			os.makedirs(save_clip_dir)
		# load images
		for cnt, img_name in enumerate(clip):
			# img_dir = os.path.join(prefix, img_name+"_leftImg8bit.png")
			img_dir = os.path.join(prefix, img_name+"_gtFine_myseg_id.png")
			with open(img_dir, 'rb') as f:
				img = Image.open(f)
				img = img.convert('RGB')
			# img_resize = img.resize(res, Image.BILINEAR)
			img_resize = img.resize(res, Image.NEAREST)
			# save_img_name = os.path.join(save_dir, img_name+"_leftImg8bit.png")
			save_img_name = os.path.join(save_dir, img_name+"_gtFine_myseg_id.png")
			img_resize.save(save_img_name)
		sys.stdout.write('\r saving {}/{} {}'.format(clip_cnt, len(clips), save_img_name))
	print()

# resize_imgs((256, 128))

def png_folder_2_avi(png_folder, out_folder, fps):
	o_f = '/'.join(out_folder.split('/')[:-1])
	if not os.path.exists(o_f):
		os.makedirs(o_f)
	files = glob.glob(png_folder+"/*.png")
	files.sort()
	frame_array = []
	for i in range(len(files)):
		filename=files[i]
		#reading each files
		img = cv2.imread(filename)
		height, width, layers = img.shape
		size = (width,height)
		
		#inserting the frames into an image array
		frame_array.append(img)
	out = cv2.VideoWriter(out_folder+".avi", cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
	for i in range(len(frame_array)):
		# writing to a image array
		out.write(frame_array[i])
	out.release()

# clip_file = "/data/linz/proj/Dataset/Cityscape/load_files/root_clip.pkl"
# with open(clip_file, 'rb') as f:
# 	clips = pickle.load(f)
# 	clips = clips['val']
# res='128x256'
# for step, clip in enumerate(clips):
# 	if step > 60:
# 		break
# 	png_folder_2_avi("cycgen/cityscape/{}/extra_wing/seg_rgb/{}".format(res, clip[0]), 
# 					"cycgen/cityscape/{}/extra_wing/seg_avi/{}".format(res, clip[0]), 17)
# 	sys.stdout.write("\r {} / 60".format(step))

def seg_id_to_rgb(seg_folder, out_folder):
	if not os.path.exists(out_folder):
		os.makedirs(out_folder)
	files = glob.glob(seg_folder+"/*.png")
	for file in files:
		name = file.split('/')[-1]
		save_name = out_folder+'/'+name
		img = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
		height, width = img.shape

		size = (width,height)
		img_rgb = vis_seg_mask(img.squeeze()[np.newaxis, :]).squeeze()
		# print(save_name)
		# print(img_rgb.shape)
		img_rgb = Image.fromarray(img_rgb)
		# cv2.imwrite(save_name,img_rgb)
		img_rgb.save(save_name)




# clip_file = "/data/linz/proj/Dataset/Cityscape/load_files/root_clip.pkl"
# with open(clip_file, 'rb') as f:
# 	clips = pickle.load(f)
# 	clips = clips['val']
# res='128x256'
# for step, clip in enumerate(clips):
# 	if step > 60:
# 		break
# 	seg_id_to_rgb("cycgen/cityscape/{}/extra_wing/seg/{}".format(res, clip[0]), 
# 					"cycgen/cityscape/{}/extra_wing/seg_rgb/{}".format(res, clip[0]))
# 	sys.stdout.write("\r {} / 60".format(step))



def combine_inter_avi():
	fps = 17*8
	clip_file = "/data/linz/proj/Dataset/Cityscape/load_files/root_clip.pkl"
	with open(clip_file, 'rb') as f:
	  clips = pickle.load(f)
	  clips = clips['val'][:61]

	gt_dir = '/data/linz/dist_proj/cycgen/cityscape/256x512/gt'
	x2_dir = '/data/linz/dist_proj/cycgen/cityscape/256x512/inter_x2'
	x4_dir = '/data/linz/dist_proj/cycgen/cityscape/256x512/inter_x4'
	x8_dir = '/data/linz/dist_proj/cycgen/cityscape/256x512/inter_x8'

	out_folder = '/data/linz/dist_proj/cycgen/cityscape/256x512/inter_comb4'

	for step, clip in enumerate(clips):
		clip_dir = clip[0]
		gt_clip_dir = os.path.join(gt_dir, clip_dir)
		gt_files = glob.glob(gt_clip_dir+'/*.png')
		gt_files.sort()
		assert len(gt_files) == 30
		for i in range(30):
			img = cv2.imread(gt_files[i])
			# height, width, layers = img.shape
			# size = (width,height)
			gt_files[i] = img

		x2_clip_dir = os.path.join(x2_dir, clip_dir)
		x2_files = glob.glob(x2_clip_dir+'/*.png')
		x2_files.sort()
		assert len(x2_files) == 59
		for i in range(59):
			img = cv2.imread(x2_files[i])
			x2_files[i] = img

		x4_clip_dir = os.path.join(x4_dir, clip_dir)
		x4_files = glob.glob(x4_clip_dir+'/*.png')
		x4_files.sort()
		assert len(x4_files) == 117
		for i in range(117):
			img = cv2.imread(x4_files[i])
			x4_files[i] = img

		x8_clip_dir = os.path.join(x8_dir, clip_dir)
		x8_files = glob.glob(x8_clip_dir+'/*.png')
		x8_files.sort()
		assert len(x8_files) == 233
		for i in range(233):
			img = cv2.imread(x8_files[i])
			x8_files[i] = img

		height, width, layers = x8_files[-1].shape
		size = (width,height)

		comb_frame_array = []
		for frame_ind in range(233):
			gt_ind = int(frame_ind / 8)
			x2_ind = int(frame_ind / 4)
			x4_ind = int(frame_ind / 2)
			x8_ind = frame_ind

			upper_img = np.concatenate([gt_files[gt_ind], x2_files[x2_ind]], axis=1)
			lower_img = np.concatenate([x4_files[x4_ind], x8_files[x8_ind]], axis=1)

			whole_img = np.concatenate([upper_img, lower_img], axis=0)
			comb_frame_array.append(whole_img)

		comb_size = (2*width, 2*height)
		save_dir = os.path.join(out_folder, clip_dir)
		save_prefix = "/".join(save_dir.split('/')[:-1])
		if not os.path.exists(save_prefix):
			os.makedirs(save_prefix)
		out = cv2.VideoWriter(save_dir+".avi", cv2.VideoWriter_fourcc(*'DIVX'), fps, comb_size)
		for i in range(len(comb_frame_array)):
			# writing to a image array
			out.write(comb_frame_array[i])
		out.release()
		sys.stdout.write("\r {}/61".format(step+1))
	print()

# combine_inter_avi()



def combine_extra_avi_v1():
	fps = 17
	clip_file = "/data/linz/proj/Dataset/Cityscape/load_files/root_clip.pkl"
	with open(clip_file, 'rb') as f:
	  clips = pickle.load(f)
	  clips = clips['val'][:61]

	# gt_dir = '/data/linz/dist_proj/cycgen/cityscape/128x256/gt'
	extra_dir = '/data/linz/dist_proj/cycgen/cityscape/128x256/extra_wing'
	extra_seg_dir = '/data/linz/dist_proj/cycgen/cityscape/128x256/extra_wing/seg_rgb'

	out_folder = '/data/linz/dist_proj/cycgen/cityscape/128x256/extra_wing/comb_v2'

	for step, clip in enumerate(clips):
		clip_dir = clip[0]

		extra_clip_dir = os.path.join(extra_dir, clip_dir)
		extra_files = glob.glob(extra_clip_dir+'/*.png')
		extra_files.sort()
		assert len(extra_files) == 28
		for i in range(28):
			img = cv2.imread(extra_files[i])
			extra_files[i] = img

		extra_seg_clip_dir = os.path.join(extra_seg_dir, clip_dir)
		extra_seg_files = glob.glob(extra_seg_clip_dir+'/*.png')
		extra_seg_files.sort()
		assert len(extra_seg_files) == 28
		for i in range(28):
			img = cv2.imread(extra_seg_files[i])
			extra_seg_files[i] = img

		height, width, layers = extra_files[-1].shape
		size = (width,height)

		comb_frame_array = []
		for frame_ind in range(28):
			img = np.concatenate([extra_files[frame_ind], extra_seg_files[frame_ind]], axis=0)
			comb_frame_array.append(img)

		comb_size = (width, 2*height)
		save_dir = os.path.join(out_folder, clip_dir)
		save_prefix = "/".join(save_dir.split('/')[:-1])
		if not os.path.exists(save_prefix):
			os.makedirs(save_prefix)
		out = cv2.VideoWriter(save_dir+".avi", cv2.VideoWriter_fourcc(*'DIVX'), fps, comb_size)
		for i in range(len(comb_frame_array)):
			# writing to a image array
			out.write(comb_frame_array[i])
		out.release()
		sys.stdout.write("\r {}/61".format(step+1))
	print()

combine_extra_avi_v1()

def combine_extra_avi():
	fps = 17/3
	clip_file = "/data/linz/proj/Dataset/Cityscape/load_files/root_clip.pkl"
	with open(clip_file, 'rb') as f:
	  clips = pickle.load(f)
	  clips = clips['val'][:61]

	gt_dir = '/data/linz/dist_proj/cycgen/cityscape/128x256/gt'
	extra_dir = '/data/linz/dist_proj/cycgen/cityscape/128x256/extra_x28'

	out_folder = '/data/linz/dist_proj/cycgen/cityscape/128x256/extra_comb'

	for step, clip in enumerate(clips):
		clip_dir = clip[0]
		gt_clip_dir = os.path.join(gt_dir, clip_dir)
		gt_files = glob.glob(gt_clip_dir+'/*.png')
		gt_files.sort()
		assert len(gt_files) == 30
		for i in range(30):
			img = cv2.imread(gt_files[i])
			# height, width, layers = img.shape
			# size = (width,height)
			gt_files[i] = img

		extra_clip_dir = os.path.join(extra_dir, clip_dir)
		extra_files = glob.glob(extra_clip_dir+'/*.png')
		extra_files.sort()
		assert len(extra_files) == 30
		for i in range(30):
			img = cv2.imread(extra_files[i])
			extra_files[i] = img

		height, width, layers = extra_files[-1].shape
		size = (width,height)

		comb_frame_array = []
		for frame_ind in range(30):
			img = np.concatenate([gt_files[frame_ind], extra_files[frame_ind]], axis=0)
			comb_frame_array.append(img)

		comb_size = (width, 2*height)
		save_dir = os.path.join(out_folder, clip_dir)
		save_prefix = "/".join(save_dir.split('/')[:-1])
		if not os.path.exists(save_prefix):
			os.makedirs(save_prefix)
		out = cv2.VideoWriter(save_dir+".avi", cv2.VideoWriter_fourcc(*'DIVX'), fps, comb_size)
		for i in range(len(comb_frame_array)):
			# writing to a image array
			out.write(comb_frame_array[i])
		out.release()
		sys.stdout.write("\r {}/61".format(step+1))
	print()

# combine_extra_avi()

