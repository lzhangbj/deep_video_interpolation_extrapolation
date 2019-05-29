from .base_options import BaseOptions

class GenOptions(BaseOptions):
	def initialize(self):
		BaseOptions.initialize(self)
		self.parser.add_argument('--model', dest='model', 
										default='MyFRRN', 
										help='model to use',
										choices=['GridNet', 'MyFRRN'])  
		# config optimization
		self.parser.add_argument('--o', dest='optimizer', 
										help='training optimizer',
										choices =['adamax','adam', 'sgd'], 
										default="adamax")
		self.parser.add_argument('--lr', dest='learning_rate', 
										help='starting learning rate',
										default=0.001, type=float)
		