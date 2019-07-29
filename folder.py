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

	def __call__(self, image, record=False, scale=1, tf=None):
		self.transforms = tf.transforms
		if record:
			assert self.crop_params is None
			assert scale == 1
		for t in self.transforms:
			if isinstance(t, transforms.RandomCrop):
				if record:
					self.crop_params = t.get_params(image, output_size=t.size)
				crop_params = ( int(scale*crop_param) for crop_param in self.crop_params )  
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
		for i, clip_boxes in enumerate(clips_boxes):
			for j,frame_boxes in enumerate(clip_boxes):
				for k,frame_box in enumerate(frame_boxes):
					if frame_box is not None:
						clips_boxes[i][j][k][0] = math.floor(clips_boxes[i][j][k][0]*150./1024.) # x1
						clips_boxes[i][j][k][1] = math.floor(clips_boxes[i][j][k][1]*150./512.)  # y1 	
						clips_boxes[i][j][k][2] = math.floor(clips_boxes[i][j][k][2]*150./1024.) # x2
						clips_boxes[i][j][k][3] = math.floor(clips_boxes[i][j][k][3]*150./512.)  # y2
						clips_boxes[i][j][k][1], clips_boxes[i][j][k][0] =  clips_boxes[i][j][k][0], clips_boxes[i][j][k][1]
						clips_boxes[i][j][k][2], clips_boxes[i][j][k][3] =  clips_boxes[i][j][k][3], clips_boxes[i][j][k][2] # y1, x1, y2, x2
						if clips_boxes[i][j][k][2] <= clips_boxes[i][j][k][0] or clips_boxes[i][j][k][3] <= clips_boxes[i][j][k][1]:
							clips_boxes[i][j][k] = None
		return clips_boxes

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
		if self.args.split == 'train':
			clip_boxes = self.bboxes[index]
		else:
			clip_boxes = torch.zeros(3,4,4)
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
							t = clip_boxes[i][j][1]
							clip_boxes[i][j][1] = 149-clip_boxes[i][j][3]
							clip_boxes[i][j][3] = 149-t

		# if isVerflip:
		# 	ori_imgs = [img.transpose(Image.FLIP_TOP_BOTTOM) for img in ori_imgs]
		# 	seg_imgs = [img.transpose(Image.FLIP_TOP_BOTTOM) for img in seg_imgs]
		# if isReverse:
		# 	ori_imgs = ori_imgs[::-1]		
		# 	seg_imgs = seg_imgs[::-1]		

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
		if self.args.split == 'train':
			# crop bboxes and transform none to [0,0,0,0]
			y1, x1, h, w = self.transform.crop_params
			for i,frame_boxes in enumerate(clip_boxes):
				for j, frame_box in enumerate(frame_boxes):
					assert frame_box is not None
					clip_boxes[i][j][0]-=y1
					clip_boxes[i][j][0] = max(0, clip_boxes[i][j][0]) 	# y1
					clip_boxes[i][j][2]-=y1
					clip_boxes[i][j][2] = min(127, clip_boxes[i][j][2]) # y2
					clip_boxes[i][j][1]-=x1
					clip_boxes[i][j][1] = max(0, clip_boxes[i][j][1]) 	# x1
					clip_boxes[i][j][3]-=x1
					clip_boxes[i][j][3] = min(127, clip_boxes[i][j][3]) # x2
					if clip_boxes[i][j][2] <= clip_boxes[i][j][0] or clip_boxes[i][j][3] <= clip_boxes[i][j][1]:
						self.transform.derecord()
						return self.__getitem__((index+randint(1,len(self.root)))%len(self.root))
				assert len(clip_boxes[i]) == 4, len(clip_boxes[i]) # control having 16 bboxes to track, 
				# clip_boxes[i].append([0,0,0,0])

			clip_boxes = torch.tensor(np.array(clip_boxes)).long() # (3, 4, 4)
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
