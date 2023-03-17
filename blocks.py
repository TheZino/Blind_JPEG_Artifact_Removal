import torch
import torch.nn as nn
import torch.nn.functional as F

############################### Custom Modules #################################

def make_layer(block, num_of_layer):
		layers = []
		for _ in range(num_of_layer):
			layers.append(block)
		return nn.Sequential(*layers)


### Residual Blocks

class Residual_Block_Spec(nn.Module):
	def __init__(self, ch=256):
		super(Residual_Block_Spec, self).__init__()

		self.conv1 = nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=3, stride=1, padding=1, bias=False)
		self.ReLU = nn.LeakyReLU(0.2, inplace=False)
		self.conv2 = nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=3, stride=1, padding=1, bias=False)

	def forward(self, x):
		identity_data = x

		output = self.ReLU(self.conv1(x))
		output = self.conv2(output)
		output = output * 0.1

		output = torch.add(output,identity_data)
		return output

### Upscaling blocks

class UpBlock(nn.Module):

	def __init__(self, in_channels, up_factor=2, kernel_size=3, stride=1, padding=1):
		super(UpBlock, self).__init__()

		self.conv = nn.Conv2d(in_channels=in_channels, out_channels=256, bias=False, kernel_size=kernel_size, stride=stride, padding=padding)
		self.ps = nn.PixelShuffle(up_factor)

	def forward(self, x):
		x = self.conv(x)
		x = self.ps(x)
		return x

### Residual in Residual Dense Blocks

def conv_block(in_nc, out_nc, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, groups=1):

	conv = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, \
			dilation=dilation, bias=bias, groups=groups)
	act = nn.LeakyReLU(negative_slope=0.2, inplace=False)

	return nn.Sequential(conv, act)

class ResidualDenseBlock_5C(nn.Module):
	"""
	Residual Dense Block
	style: 5 convs
	The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
	"""

	def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True):
		super(ResidualDenseBlock_5C, self).__init__()
		# gc: growth channel, i.e. intermediate channels
		self.conv1 = conv_block(nc, gc, kernel_size, stride, bias=bias)
		self.conv2 = conv_block(nc+gc, gc, kernel_size, stride, bias=bias)
		self.conv3 = conv_block(nc+2*gc, gc, kernel_size, stride, bias=bias)
		self.conv4 = conv_block(nc+3*gc, gc, kernel_size, stride, bias=bias)

		self.conv5 = conv_block(nc+4*gc, nc, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, groups=1)

	def forward(self, x):
		x1 = self.conv1(x)
		x2 = self.conv2(torch.cat((x, x1), 1))
		x3 = self.conv3(torch.cat((x, x1, x2), 1))
		x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
		x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
		return x5.mul(0.2) + x


class RRDB(nn.Module):
	"""
	Residual in Residual Dense Block
	"""

	def __init__(self, nc, kernel_size=3, gc=32, stride=1, bias=True):
		super(RRDB, self).__init__()
		self.RDB1 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias)
		self.RDB2 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias)
		self.RDB3 = ResidualDenseBlock_5C(nc, kernel_size, gc, stride, bias)

	def forward(self, x):
		out = self.RDB1(x)
		out = self.RDB2(out)
		out = self.RDB3(out)
		return out.mul(0.2) + x

### Basic Blocks

class CBLR_block(nn.Module):

	def __init__(self, in_channels, out_channels, **kwargs):
		super(CBLR_block, self).__init__()

		self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
		self.bn = nn.BatchNorm2d(out_channels)

	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		return F.leaky_relu(x, negative_slope=0.2, inplace=False)
