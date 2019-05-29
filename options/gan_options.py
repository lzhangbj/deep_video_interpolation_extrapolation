from .base_options import BaseOptions

class GANOptions(BaseOptions):
		def initialize(self):
			BaseOptions.initialize(self)
			self.parser.add_argument('--model', dest='model', 
											default='GAN', 
											help='model to use',
											choices=['GAN'])  
			self.parser.add_argument('--netG', dest='netG', 
											default='MyFRRN', 
											help='model to use',
											choices=['GridNet', 'MyFRRN'])  
			self.parser.add_argument('--netD', dest='netD', 
											default='multi_scale', 
											help='model to use',
											choices=['multi_scale'])
			self.parser.add_argument('--numD', dest='num_D', 
											default=3, 
											help='number of discriminator',
											type=int)
			self.parser.add_argument('--n_layer_D', dest='n_layer_D', 
											default=2, 
											help='number of discriminator layers',
											type=int)
			self.parser.add_argument('--oG', dest='optG', 
											help='training optimizer',
											choices =['adamax','adam', 'sgd'], 
											default="adamax")
			self.parser.add_argument('--oD', dest='optD', 
											help='training optimizer',
											choices =['adamax','adam', 'sgd'], 
											default="sgd")
			self.parser.add_argument('--lrG', dest='lr_G', 
											help='starting learning rate',
											default=0.001, type=float)
			self.parser.add_argument('--lrD', dest='lr_D', 
											help='starting learning rate',
											default=0.001, type=float)
			self.parser.add_argument('--adv_w', dest='adv_weight',
											help='training optimizer loss weigh of gdl',
											type=float,
											default=1)
			self.parser.add_argument('--adv_feat_w', dest='adv_feat_weight',
											help='training optimizer loss weigh of gdl',
											type=float,
											default=1)
			self.parser.add_argument('--d_w', dest='d_weight',
											help='training optimizer loss weigh of gdl',
											type=float,
											default=10)

