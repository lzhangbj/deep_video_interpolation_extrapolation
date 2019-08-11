import torch.utils.data as data
from PIL import Image
import cv2
import os
import os.path
import sys
from random import randint
import torch
import torchvision.transforms as transforms
import numpy as np
import math

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', 'webp']

def pil_loader_rgb(path):
	# open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
	with open(path, 'rb') as f:
		img = Image.open(f)
		return img.convert('RGB')

def pil_loader_seg(path):
	# open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
	with open(path, 'rb') as f:
		img = Image.open(f)
		return img.convert('L')


def rgb_load(img_list):
	return [pil_loader_rgb(i) for i in img_list]

def seg_load(seg_list):
	l = []
	for i in seg_list:
		seg_img = pil_loader_seg(i)
		l.append(seg_img)
	return l

class Transform(object):
	def __init__(self):
		super(Transform, self).__init__()
		# self.transforms = tf.transforms
		self.crop_params = None

	def __call__(self, image, record=False, scale=1, tf=None, pre_crop_params=None):
		'''
			pre_crop_params: (h_min, w_min, height, width)
		'''
		self.transforms = tf.transforms
		if record:
			assert self.crop_params is None
			assert scale == 1
		for t in self.transforms:
			if isinstance(t, transforms.RandomCrop):
				assert not (pre_crop_params is not None and record), 'pre_crop_params and record can not be both used'
				if pre_crop_params is None:
					if record:
						self.crop_params = t.get_params(image, output_size=t.size)
					crop_params = ( int(scale*crop_param) for crop_param in self.crop_params )  
				else:
					crop_params = pre_crop_params
				image = transforms.functional.crop(image, *crop_params)
			image = t(image)
		return image

	def derecord(self):
		self.crop_params = None

	def __repr__(self):
		format_string = self.__class__.__name__ + '('
		for t in self.transforms:
			format_string += '\n'
			format_string += '    {0}'.format(t)
		format_string += '\n)'
		return format_string

class DatasetFolder(data.Dataset):
	def __init__(self, args, root, transform=None, target_transform=None, bboxes=None):
		self.root = root
		self.tfs = transform
		self.transform = Transform()
		self.target_transform = target_transform
		self.args = args
		if self.args.dataset=='cityscape':
			if self.args.split == 'train':
				self.img_dir = args.img_dir if args.img_dir is not None else "/data/linz/proj/Dataset/Cityscape/leftImg_sequence/leftImg8bit_sequence_150x150/"
				self.seg_dir = args.seg_dir if args.seg_dir is not None else "/data/linz/proj/Dataset/Cityscape/leftImg_sequence/gtFine_myseg_id_sequence_150x150/"
				self.bboxes = self.convert(bboxes)
			elif self.args.split == 'val':
				self.img_dir = args.img_dir if args.img_dir is not None else "/data/linz/proj/Dataset/Cityscape/leftImg_sequence/leftImg8bit_sequence_128x128/"
				self.seg_dir = args.seg_dir if args.seg_dir is not None else "/data/linz/proj/Dataset/Cityscape/leftImg_sequence/gtFine_myseg_id_sequence_128x128/"
				if bboxes is not None:
					self.bboxes = self.convert(bboxes)
			self.img_ext = "_leftImg8bit.png"
			self.seg_ext = "_gtFine_myseg_id.png"
			self.edge_dir = "/data/linz/proj/Dataset/Cityscape/leftImg_sequence/edges256/"
			self.edge_ext = "_edge.png"
			self.disparity_dir = '/data/linz/proj/Dataset/Cityscape/disparity_sequence/'
			self.disparity_ext = '_disparity.png'
		elif self.args.dataset == 'ucf101':
			self.img_dir = "/data/linz/proj/Dataset/CyclicGen-master/ucf101_interp_ours/"
			self.img_ext = ".png"
		elif self.args.dataset == 'vimeo':
			self.img_dir = "/data/linz/proj/Dataset/vimeo_triplet/sequences/"
			self.img_ext = ".png"
		self.n_classes = 20
		self.vid_len = len(self.root[0])

	def convert(self,clips_boxes):
		img_size = 150 if self.args.split == 'train' else 128
		for i, clip_boxes in enumerate(clips_boxes):
			for j,frame_boxes in enumerate(clip_boxes):
				for k,frame_box in enumerate(frame_boxes):
					if frame_box is not None:
						clips_boxes[i][j][k][1] = math.floor(clips_boxes[i][j][k][1]*img_size/1024.) # x1
						clips_boxes[i][j][k][2] = math.floor(clips_boxes[i][j][k][2]*img_size/512.)  # y1 	
						clips_boxes[i][j][k][3] = math.floor(clips_boxes[i][j][k][3]*img_size/1024.) # x2
						clips_boxes[i][j][k][4] = math.floor(clips_boxes[i][j][k][4]*img_size/512.)  # y2
						clips_boxes[i][j][k][2], clips_boxes[i][j][k][1] =  clips_boxes[i][j][k][1], clips_boxes[i][j][k][2]
						clips_boxes[i][j][k][4], clips_boxes[i][j][k][3] =  clips_boxes[i][j][k][3], clips_boxes[i][j][k][4] # y1, x1, y2, x2
						if clips_boxes[i][j][k][3] <= clips_boxes[i][j][k][1] or clips_boxes[i][j][k][4] <= clips_boxes[i][j][k][2]:
							clips_boxes[i][j][k] = None
				assert len(frame_boxes) == self.args.num_track_per_img, len(frame_boxes)
		return clips_boxes

	def get_seq_crop_params(self):
		h_interval = np.random.randint(150-128)
		w_interval = np.random.randint(150-128)
		h_dir = np.random.randint(2)
		w_dir = np.random.randint(2)
		mid_h1 = np.random.randint(h_interval//2, 150-128-h_interval//2)
		mid_w1 = np.random.randint(w_interval//2, 150-128-w_interval//2)
		if h_dir == 1: 	# left to right
			for_h1 = mid_h1-h_interval//2
			back_h1 = mid_h1+h_interval//2
		else:			# right to left
			for_h1 = mid_h1+h_interval//2
			back_h1 = mid_h1-h_interval//2

		if w_dir == 1: 	# top to bottom 
			for_w1 = mid_w1-w_interval//2
			back_w1 = mid_w1+w_interval//2
		else:			# bot to top
			for_w1 = mid_w1+w_interval//2
			back_w1 = mid_w1-w_interval//2

		assert for_h1 >= 0 and for_h1 < 150-128
		assert mid_h1 >= 0 and mid_h1 < 150-128
		assert back_h1 >= 0 and back_h1 < 150-128
		return (for_h1, for_w1, 128, 128), (mid_h1, mid_w1, 128, 128), (back_h1, back_w1, 128, 128)

	def __getitem__(self, index):
		if self.args.dataset != 'vimeo':
			img_files = [self.img_dir+self.root[index][i]+self.img_ext for i in range(self.vid_len)]
		else:
			assert self.args.mode == 'xx2x'
			img_files = [self.img_dir+self.root[index]+ "/im{}".format(i+1) + self.img_ext for i in range(self.vid_len)]
		ori_imgs = rgb_load(img_files)
		if self.args.dataset == 'cityscape':
			seg_files = [self.seg_dir+self.root[index][i]+self.seg_ext for i in range(self.vid_len)]
			seg_imgs = seg_load(seg_files)
			seg_imgs_ori = []
		try:
			clip_boxes = self.bboxes[index].copy()
		except:
			clip_boxes = torch.zeros(3,self.args.num_track_per_img,4)
		isHorflip = randint(0,2)
		# isVerflip = randint(0,2)
		# isReverse = randint(0,2)
		if isHorflip and self.args.split == 'train':
			ori_imgs = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in ori_imgs]
			seg_imgs = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in seg_imgs]
			if self.args.split == 'train':
				assert len(clip_boxes) == 3
				assert len(clip_boxes[0]) == len(clip_boxes[1]) and len(clip_boxes[1]) == len(clip_boxes[2])
				for i,frame_boxes in enumerate(clip_boxes):
					for j, frame_box in enumerate(frame_boxes):
						if frame_box is not None:
							t = clip_boxes[i][j][2]
							clip_boxes[i][j][2] = 149-clip_boxes[i][j][4]
							clip_boxes[i][j][4] = 149-t

		# sample crop images
		if self.args.split == 'train':
			seq_crop_params = self.get_seq_crop_params()

			for i in range(self.vid_len):
				ori_imgs[i] = transforms.functional.normalize( 
										transforms.functional.to_tensor(
												self.transform(ori_imgs[i], tf=self.tfs[0], pre_crop_params=seq_crop_params[i])
										),  (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
									)
				if self.args.dataset == 'cityscape':
					np_seg = np.array(self.transform(seg_imgs[i], scale=1,\
										 tf=self.tfs[1], pre_crop_params=seq_crop_params[i]))
					np_seg = np.eye(20)[np_seg]
					seg_imgs[i] = torch.from_numpy(
												np.transpose(
													np_seg, (2,0,1)
													)
												).float() #* 2 - 1
		else:
			for i in range(self.vid_len):
				ori_imgs[i] = transforms.functional.normalize( 
										transforms.functional.to_tensor(
												self.transform(ori_imgs[i], record=(i==0), tf=self.tfs[0])
										),  (0.5, 0.5, 0.5),(0.5, 0.5, 0.5)
									)
				if self.args.dataset == 'cityscape':
					np_seg = np.array(self.transform(seg_imgs[i], scale= 1,\
										 tf=self.tfs[1]))
					np_seg = np.eye(20)[np_seg]
					seg_imgs[i] = torch.from_numpy(
												np.transpose(
													np_seg, (2,0,1)
													)
												).float() #* 2 - 1

		# if not (not self.args.track_gen and self.args.split=='val'):
		new_clip_boxes = [[], [], []]
		assert 	len(clip_boxes[0]) == self.args.num_track_per_img and \
				len(clip_boxes[1]) == self.args.num_track_per_img and \
				len(clip_boxes[2]) == self.args.num_track_per_img, [len(clip_boxes[0]),len(clip_boxes[1]),len(clip_boxes[2])]

		for j in range(self.args.num_track_per_img):
			for i in range(3):
				bbox = clip_boxes[i][j]
				if bbox is None:
					break
				bbox = bbox.copy()
				if self.args.split == 'train':
					y1, x1, h, w = seq_crop_params[i]
				else:
					y1=0
					x1=0
				bbox[1]-=y1
				bbox[1] = max(0, bbox[1]) 	# y1
				bbox[3]-=y1
				bbox[3] = min(127, bbox[3]) # y2
				bbox[2]-=x1
				bbox[2] = max(0, bbox[2]) 	# x1
				bbox[4]-=x1
				bbox[4] = min(127,bbox[4]) 	# x2
				clip_boxes[i][j] = bbox
				if bbox[3] <= bbox[1] or bbox[4] <= bbox[2]:
					break
				if i == 2:
					assert clip_boxes[2][j] == bbox
					for t in range(3):
						assert clip_boxes[t][j][3] > clip_boxes[t][j][1] and clip_boxes[t][j][4] > clip_boxes[t][j][2], \
										[clip_boxes[0][j],clip_boxes[1][j],clip_boxes[2][j]]
						new_clip_boxes[t].append(clip_boxes[t][j])
		for i in new_clip_boxes:
			for j in i:
				assert j[3] > j[1] and j[4] > j[2] and j[0] > 0 and j[0] < 1, new_clip_boxes

		if len(new_clip_boxes[1]) == 0:
			self.transform.derecord()
			return self.__getitem__((index+randint(1,len(self.root)))%len(self.root))
		elif len(new_clip_boxes[1]) < self.args.num_track_per_img:
			existed_track_num = len(new_clip_boxes[1])
			while len(new_clip_boxes[1]) < self.args.num_track_per_img:
				rand_ind = np.random.randint(existed_track_num)
				for i in range(3):
					new_clip_boxes[i].append(new_clip_boxes[i][rand_ind].copy())
		clip_boxes = new_clip_boxes
		for j in range(self.args.num_track_per_img):
			for i in range(3):
				bbox = clip_boxes[i][j] 
				assert  bbox[3] >= bbox[1] and bbox[4] >= bbox[2], clip_boxes

		clip_boxes = torch.tensor(np.array(clip_boxes)).float() # (3, 10, 4)
		self.transform.derecord()
		if self.args.split == 'train':
			if not self.check_clip_boxes(clip_boxes):
				print('meet none_have bboxes')
				return self.__getitem__(index+1)
		if self.args.dataset == 'cityscape':
			return_dict = {}
			for i in range(self.vid_len):
				return_dict['frame'+str(i+1)] = ori_imgs[i]		
				return_dict['seg'+str(i+1)] = seg_imgs[i]	
				return_dict['bboxes'] = clip_boxes	
			return return_dict
		else:
			return {'frame1':ori_imgs[0],
					'frame2':ori_imgs[1],
					'frame3':ori_imgs[2],
					'seg1'  :torch.zeros(1,1),
					'seg2'  :torch.zeros(1,1),
					'seg3'  :torch.zeros(1,1)}

	def __len__(self):
		return len(self.root)

	def check_clip_boxes(self, clip_boxes):
		if clip_boxes[0].sum() == 0 and clip_boxes[2].sum() ==0:
			return False
		return True

	def __repr__(self):
		fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
		fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
		fmt_str += '    Root Location: {}\n'.format(self.root)
		tmp = '    Transforms (if any): '
		fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
		tmp = '    Target Transforms (if any): '
		fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
		return fmt_str

class ImageFolder(DatasetFolder):
	def __init__(self, args, root, transform=None, bboxes=None):
		super(ImageFolder, self).__init__(args, root, transform, bboxes=bboxes)
