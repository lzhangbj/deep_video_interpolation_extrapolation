import torchvision.transforms as transforms
from folder import ImageFolder
from PIL import Image
import pickle
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
		clip_file = "/data/linz/proj/Dataset/Cityscape/load_files/int_{}_len_3_extra_lsclip.pkl".format(args.interval)
		with open(clip_file, 'rb') as f:
			load_f = pickle.load(f)
			clips_train_file = load_f['train'] 
			clips_val_file = load_f['val'] 
		train_dataset = ImageFolder(args, clips_train_file,
												transform=transforms.Compose([
													transforms.Resize((150,300), interpolation=Image.NEAREST),
													transforms.RandomCrop((128,256))
												])
											)
		val_dataset = ImageFolder(args, clips_val_file,
												transform=transforms.Compose([
													transforms.Resize((128,256), interpolation=Image.NEAREST)
												])
											)
		# val_dataset = ImageFolder(args, clips_val_file,transform=None)
	else:
		raise Exception('Invalid dataset %s' % args.dataset)
	
	return train_dataset, val_dataset


