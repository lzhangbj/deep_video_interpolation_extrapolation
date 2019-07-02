import torch
import torch.nn as nn


class my_resnet101(nn.Module):
	def __init__(self, resnet101):
		super(my_resnet101, self).__init__()
		self.resnet101 = resnet101

	def forward(self, x):
		x = self.resnet101.conv1(x)
		x = self.resnet101.bn1(x)
		x = self.resnet101.relu(x)
		x = self.resnet101.maxpool(x)

		x = self.resnet101.layer1(x)
		x1 = self.resnet101.layer2(x)
		x2 = self.resnet101.layer3(x1)
		x3 = self.resnet101.layer4(x2)

		return x1, x2, x3
