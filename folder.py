import torch.utils.data as data
from PIL import Image
import cv2
import json
import os
import os.path
import sys
from random import randint
import torch
import torchvision.transforms as transforms
from itertools import groupby
from operator import itemgetter
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
        # seg_np = np.array(seg_img)
        # l.append(Image.fromarray(np.eye(20)[seg_np], mode='L'))
    return l




class Transform(object):
    def __init__(self, tf):
        super(Transform, self).__init__()
        self.transforms = tf.transforms
        self.crop_params = None

    def __call__(self, image, record=False):
        if record:
            assert self.crop_params is None
        for t in self.transforms:
            if isinstance(t, transforms.RandomCrop):
                if record:
                    self.crop_params = t.get_params(image, output_size=t.size)
                image = transforms.functional.crop(image, *self.crop_params)
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
        self.transform = Transform(transform)
        self.target_transform = target_transform

        self.img_dir = args.img_dir if args.img_dir is not None else "/data/linz/proj/Dataset/Cityscape/leftImg_sequence/leftImg8bit_sequence/"
        self.img_ext = "_leftImg8bit.png"
        self.seg_dir = args.seg_dir if args.seg_dir is not None else "/data/linz/proj/Dataset/Cityscape/leftImg_sequence/gtFine_myseg_id_sequence/"
        self.seg_ext = "_gtFine_myseg_id.png"
        self.edge_dir = "/data/linz/proj/Dataset/Cityscape/leftImg_sequence/edges256/"
        self.edge_ext = "_edge.png"
        self.disparity_dir = '/data/linz/proj/Dataset/Cityscape/disparity_sequence/'
        self.disparity_ext = '_disparity.png'
        self.n_classes = 20
        self.vid_len = len(self.root[0])

    def __getitem__(self, index):
        img_files = [self.img_dir+self.root[index][i]+self.img_ext for i in range(self.vid_len)]
        seg_files = [self.seg_dir+self.root[index][i]+self.seg_ext for i in range(self.vid_len)]
        disparity_files = [self.disparity_dir+self.root[index][i]+self.disparity_ext for i in range(self.vid_len)]

        ori_imgs = rgb_load(img_files)
        seg_imgs = seg_load(seg_files)
        edge = pil_loader_seg(self.edge_dir+self.root[index][1]+self.edge_ext)
        disparity_imgs = [pil_loader_seg(disparity_files[i]) for i in range(self.vid_len)]
        seg_imgs_ori = []

        seg_fg_masks = []
        seg_bg_masks = []

        for i in range(self.vid_len):
            # ori_imgs[i] = transforms.functional.normalize( 
            ori_imgs[i] = transforms.functional.to_tensor(
                                    self.transform(ori_imgs[i], i==0)
                                )#,  (0.5, 0.5, 0.5),(0.5, 0.5, 0.5)
                                # )
            np_seg = np.array(self.transform(seg_imgs[i]))
            fg_mask = np.logical_and(np_seg >= 11, np_seg <= 18)[np.newaxis, :, :].astype(np.float32)
            bg_mask = 1-fg_mask

            seg_fg_masks.append(torch.from_numpy(fg_mask).float())
            seg_bg_masks.append(torch.from_numpy(bg_mask).float())

            # np_seg_ori = np_seg[:, :, np.newaxis]
            # seg_imgs_ori.append(torch.from_numpy(
            #                             np.transpose(
            #                                 np_seg_ori, (2,0,1)
            #                                 )
            #                             ).float())
            np_seg = np.eye(20)[np_seg]
            seg_imgs[i] = torch.from_numpy(
                                        np.transpose(
                                            np_seg, (2,0,1)
                                            )
                                        ).float()  #* 2 - 1
            disparity_imgs[i] = torch.from_numpy(
                                        np.array(self.transform(disparity_imgs[i]))[np.newaxis, :, :]
                                    ).float()


        # t = np.array(self.transform(edge))[np.newaxis, :, :]/255.
        # print(t.shape)
        # edge = torch.from_numpy(t).float()

        self.transform.derecord()

        frames = torch.stack(ori_imgs, dim=0)
        segs = torch.stack(seg_imgs, dim=0)
        disparities = torch.stack(disparity_imgs, dim=0) 
        fg_masks = torch.stack(seg_fg_masks, dim=0)
        bg_masks = torch.stack(seg_bg_masks, dim=0)
        return {'frames':frames,
                'segs': segs,
                'disparities': disparities,
                'fg_masks': fg_masks,
                'bg_masks': bg_masks}
        # return {'frame1':ori_imgs[0],
        #         'frame2':ori_imgs[1],
        #         'frame3':ori_imgs[2],
        #         'seg1'  :seg_imgs[0],
        #         'seg2'  :seg_imgs[1],
        #         'seg3'  :seg_imgs[2],
        #         'edge'  :edge}            
                # 'seg_id1'  :seg_imgs_ori[0],
                # 'seg_id2'  :seg_imgs_ori[1],
                # 'seg_id3'  :seg_imgs_ori[2]}

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
