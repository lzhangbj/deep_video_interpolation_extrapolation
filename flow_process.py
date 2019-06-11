#import h5py
import pickle
import json
import shutil
import numpy as np
import numpy.random as nr
import sys
import os
import collections

import glob


def record_eff_img(store_dir):
	types = ['train', 'val']
	store_dict = {'train':[],
					'val':[]	}
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

filter_load_file('/data/linz/proj/Dataset/Cityscape/load_files/int_1_len_3_extra_lsclip.pkl',
				'/data/linz/proj/Dataset/Cityscape/optical_flow/effec_optical_flow.pkl',
				'/data/linz/proj/Dataset/Cityscape/load_files/effec_flow_int_1_len_3_extra_lsclip.pkl')


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

# def find_outlier_region()