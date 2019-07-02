import torchvision.transforms as transforms
from folder import ImageFolder
from PIL import Image
# from cityscape_utils import *
'''
	input:			 dataset name(str)

	return np data:	
		n_classes	: int

		train_imgs	: (n_t, h, w, 3)
		train_segs	: (n_t, h, w)
		train_masks	: (n_t, h, w)  missing region is 0, known region is 1 

		val_imgs	: (n_v, h, w, 3)
		val_segs	: (n_v, h, w)
		val_masks	: (n_v, h, w)
'''

def get_dataset(args):
	### explicitly set flip = True #######
	if args.dataset == "cityscape":
		clip_file = "/data/linz/proj/Dataset/Cityscape/load_files/int_{}_len_3_extra_lsclip.pkl".format(int(args.interval))
		if args.vid_length != 1:
			clip_file = "/data/linz/proj/Dataset/Cityscape/load_files/int_{}_len_{}_extra_lsclip.pkl".format(int(args.interval), args.vid_length+2)
		if args.effec_flow:
			clip_file = "/data/linz/proj/Dataset/Cityscape/load_files/effec_flow_int_{}_len_3_extra_lsclip.pkl".format(int(args.interval))
		with open(clip_file, 'rb') as f:
			import pickle
			load_f = pickle.load(f)
			clips_train_file = load_f['train'] 
			clips_val_file = load_f['val'] 

		if args.high_res:
			# re_size = (300, 600)
			crop_size = (256, 512)
		else:
			# re_size = (150, 300)
			crop_size = (128, 256)
		# train 
		tfs = []
		tfs.append(transforms.Compose([		#transforms.Resize(re_size, interpolation=Image.BILINEAR),
											transforms.RandomCrop(crop_size)]))
		# tfs.append(transforms.Compose([		transforms.Resize((150, 300), interpolation=Image.NEAREST),
		# 											transforms.RandomCrop((128, 256))	]))
		tfs.append(transforms.Compose([		#transforms.Resize((150, 300), interpolation=Image.NEAREST),
													transforms.RandomCrop(crop_size)	]))

		train_dataset = ImageFolder(args, clips_train_file, transform=tfs)		

		# val
		tfs = []
		tfs.append(transforms.Compose([		#transforms.Resize(crop_size, interpolation=Image.BILINEAR)
												]))
		tfs.append(transforms.Compose([		#transforms.Resize((128, 256), interpolation=Image.NEAREST)
												]))

		val_dataset   = ImageFolder(args, clips_val_file, 
												transform=tfs
											)
	elif args.dataset == "ucf101":
		clip_file = "/data/linz/proj/Dataset/CyclicGen-master/UCF101_test_root_clip.pkl"
		with open(clip_file, 'rb') as f:
			import pickle
			load_f = pickle.load(f)
			clips_val_file = load_f['test'] 
		re_size   = (256, 256)
		crop_size = (256, 256)	
		train_dataset = None
		# val
		tfs = []
		tfs.append(transforms.Compose([		transforms.Resize(crop_size, interpolation=Image.BILINEAR)
												]))
		tfs.append(transforms.Compose([		transforms.Resize((256, 256), interpolation=Image.NEAREST)
												]))

		val_dataset   = ImageFolder(args, clips_val_file, 
												transform=tfs
											)
		# val_dataset = ImageFolder(args, clips_val_file,transform=None)
	elif args.dataset == 'vimeo':
		clip_train_file = '/data/linz/proj/Dataset/vimeo_triplet/tri_trainlist.txt'
		clip_val_file = '/data/linz/proj/Dataset/vimeo_triplet/tri_testlist.txt'
		clips_file = {'train':[],
						'val':[]}
		with open(clip_train_file, 'r') as f:
			for line in f:
				line = line.strip()
				if len(line) < 4:
					break
				clips_file['train'].append(line)
		with open(clip_val_file, 'r') as f:
			for line in f:
				line = line.strip()
				if len(line) < 4:
					break
				clips_file['val'].append(line)

		# crop_size = (128, 224)
		# train 
		tfs = []
		tfs.append(transforms.Compose([		#transforms.Resize(re_size, interpolation=Image.BILINEAR),
											# transforms.RandomCrop(crop_size)
											]))
		# tfs.append(transforms.Compose([		transforms.Resize((150, 300), interpolation=Image.NEAREST),
		# 											transforms.RandomCrop((128, 256))	]))
		tfs.append(transforms.Compose([		#transforms.Resize((150, 300), interpolation=Image.NEAREST),
													# transforms.RandomCrop(crop_size)	
													]))

		train_dataset = ImageFolder(args, clips_file['train'], transform=tfs)		

		# val
		tfs = []
		tfs.append(transforms.Compose([		#transforms.Resize(crop_size, interpolation=Image.BILINEAR)
												]))
		tfs.append(transforms.Compose([		#transforms.Resize((128, 256), interpolation=Image.NEAREST)
												]))

		val_dataset   = ImageFolder(args, clips_file['val'], 
												transform=tfs
											)



	else:
		raise Exception('Invalid dataset %s' % args.dataset)
	
	return train_dataset, val_dataset


