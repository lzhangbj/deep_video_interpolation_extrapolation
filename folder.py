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
	def __init__(self, args, root, transform=None, target_transform=None):
		self.root = root
		self.tfs = transform
		self.transform = Transform()
		self.target_transform = target_transform
		self.args = args
		if self.args.dataset=='cityscape':
			if self.args.split == 'train':
				self.img_dir = args.img_dir if args.img_dir is not None else "/data/linz/proj/Dataset/Cityscape/leftImg_sequence/leftImg8bit_sequence_150x300/"
				self.seg_dir = args.seg_dir if args.seg_dir is not None else "/data/linz/proj/Dataset/Cityscape/leftImg_sequence/gtFine_myseg_id_sequence_150x300/"
			elif self.args.split == 'val':
				self.img_dir = args.img_dir if args.img_dir is not None else "/data/linz/proj/Dataset/Cityscape/leftImg_sequence/leftImg8bit_sequence_128x256/"
				self.seg_dir = args.seg_dir if args.seg_dir is not None else "/data/linz/proj/Dataset/Cityscape/leftImg_sequence/gtFine_myseg_id_sequence_128x256/"
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

	def __getitem__(self, index):
		if self.args.dataset != 'vimeo':
			img_files = [self.img_dir+self.root[index][i]+self.img_ext for i in range(self.vid_len)]
		else:
			assert self.args.mode == 'xx2x'
			img_files = [self.img_dir+self.root[index]+ "/im{}".format(i+1) + self.img_ext for i in range(self.vid_len)]
		ori_imgs = rgb_load(img_files)
		# if self.args.dataset == 'vimeo': # random flip


		if self.args.dataset == 'cityscape':
			seg_files = [self.seg_dir+self.root[index][i]+self.seg_ext for i in range(self.vid_len)]
			seg_imgs = seg_load(seg_files)
			seg_imgs_ori = []

			seg_fg_masks = []
			seg_bg_masks = []

		isHorflip = randint(0,2)
		isVerflip = randint(0,2)
		isReverse = randint(0,2)
		if isHorflip:
			ori_imgs = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in ori_imgs]
			seg_imgs = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in seg_imgs]
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
				np_seg = np.array(self.transform(seg_imgs[i], scale= 0.5 if self.args.high_res else 1,\
									 tf=self.tfs[1]))
				fg_mask = np.logical_and(np_seg >= 11, np_seg <= 18)[np.newaxis, :, :].astype(np.float32)
				# bg_mask = 1-fg_mask
				seg_fg_masks.append(torch.from_numpy(fg_mask).float())

				np_seg = np.eye(20)[np_seg]
				seg_imgs[i] = torch.from_numpy(
											np.transpose(
												np_seg, (2,0,1)
												)
											).float() #* 2 - 1
		self.transform.derecord()
		# return ori_imgs[0]
		if self.args.dataset == 'cityscape':
			return {'frame1':ori_imgs[0],
					'frame2':ori_imgs[1],
					'frame3':ori_imgs[2],
					'frame4':ori_imgs[3] if self.vid_len > 3 else torch.zeros(1,1),
					'frame5':ori_imgs[4] if self.vid_len > 4 else torch.zeros(1,1),
					'seg1'  :seg_imgs[0],
					'seg2'  :seg_imgs[1],
					'seg3'  :seg_imgs[2],
					'fg_mask1': seg_fg_masks[0],
					'fg_mask2': seg_fg_masks[1],
					'fg_mask3': seg_fg_masks[2]}
		else:
			return {'frame1':ori_imgs[0],
					'frame2':ori_imgs[1],
					'frame3':ori_imgs[2],
					'seg1'  :torch.zeros(1,1),
					'seg2'  :torch.zeros(1,1),
					'seg3'  :torch.zeros(1,1),
					'fg_mask1': torch.zeros(1,1),
					'fg_mask2': torch.zeros(1,1),
					'fg_mask3': torch.zeros(1,1)}

	def __len__(self):
		return len(self.root)

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
	def __init__(self, args, root, transform=None):
		super(ImageFolder, self).__init__(args, root, transform)
