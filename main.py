import argparse
import datetime
import logging
import pathlib
import random
import socket
import sys

import torch
import torch.distributed as dist 
import torch.multiprocessing as mp 
import numpy as np 

from runners.trainer import Trainer

from runners.ganer import GANer
from options.gan_options import GANOptions
from options.generator_options import GenOptions
from options.options import Options
from subprocess import call

import warnings
warnings.filterwarnings("ignore")


def get_exp_path(args):
	'''Retrun new experiment path.'''
	dir = "{}_{}_{}_{}".format(args.model, args.mode, args.syn_type, args.session)
	return '{}/{}_{}'.format(args.save_dir, dir, datetime.datetime.now().strftime('%m-%d-%H:%M:%S'))


def get_logger(path, rank=None):
	'''Get logger for experiment.'''
	logger = logging.getLogger(__name__)
	logger.setLevel(logging.INFO)

	if rank is None:
		formatter = logging.Formatter('%(asctime)s-%(message)s')
	else:
		formatter = logging.Formatter('%(asctime)s - [worker '
			+ str(rank) +'] - %(message)s')
	
	# stdout log
	handler = logging.StreamHandler(sys.stdout)
	handler.setLevel(logging.INFO)
	handler.setFormatter(formatter)
	logger.addHandler(handler)

	# file log
	handler = logging.FileHandler(path)
	handler.setLevel(logging.INFO)
	handler.setFormatter(formatter)
	logger.addHandler(handler)
	return logger


def worker(rank, args):
	if not args.val:
		logger = get_logger(args.path + '/experiment.log',
						rank) # process specific logger
	else:
		logger = get_logger(args.path + '/experiment_val.log',
						rank) # process specific logger
	args.logger = logger
	args.rank = rank
	dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:%d' % args.port,
		world_size=args.gpus, rank=args.rank)

	# seed
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)

	if args.runner =='gen':
		trainer = Trainer(args)
	elif args.runner =='gan':
		trainer = GANer(args)
	else:
		raise Exception("speficied runner does not exist !")

	if args.val:
		if args.checkepoch_range:
			for i in range(args.checkepoch_low, args.checkepoch_up+1):
				args.checkepoch = i
				trainer.load_checkpoint()
				trainer.validate()
		else:
			trainer.validate()
	else:
		for epoch in range(trainer.epoch-1, args.epochs):
			trainer.set_epoch(epoch)
			trainer.train()
			# metrics = trainer.validate() disable validation

			if args.rank == 0:	# gpu id
				trainer.save_checkpoint()


def main():
	parser = Options()
	args = parser.parse()
	
	# exp path
	args.path = get_exp_path(args)
	if args.resume or args.val or args.load_G:
		args.path = args.load_dir
	else:
		pathlib.Path(args.path).mkdir(parents=True, exist_ok=False)
		(pathlib.Path(args.path) / 'checkpoint').mkdir(parents=True, exist_ok=False)

	# find free port
	if args.port is None:
		with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
			s.bind(('', 0))
			s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
			args.port = int(s.getsockname()[1])

	# logger
	if args.val:
		logger = get_logger(args.path + '/experiment_val.log') if args.interval == 2 else \
					 get_logger(args.path + '/experiment_val_int_1.log')
	else:
		logger = get_logger(args.path + '/experiment.log')
	logger.info('Start of experiment')
	logger.info('=========== Initilized logger =============')
	logger.info('\n\t' + '\n\t'.join('%s: %s' % (k, str(v))
		for k, v in sorted(dict(vars(args)).items())))
	
	# distributed training
	args.gpus = torch.cuda.device_count()
	logger.info('Total number of gpus: %d' % args.gpus)
	mp.spawn(worker, args=(args,), nprocs=args.gpus)

if __name__ == '__main__':
	mp.set_start_method('spawn', force=True)
	main()


# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3,4 python main.py  --disp_interval 100 --mode xs2xs --syn_type inter  --load_dir log/GAN_xs2xs_inter_1_05-09-21:47:39 --bs 48 --nw 8  --checksession 1 --s 0  --checkepoch 20 --checkpoint 1611  gan --d_w 10 --netD multi_scale_img --load_G --load_GANG --lrD 0.001 --n_layer_D 4