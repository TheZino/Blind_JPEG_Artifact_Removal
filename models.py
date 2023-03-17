import blocks as B
import torch
import torch.nn as nn
from torchvision import models

############################# Net Architecture #################################

class AR_Net(nn.Module):

	def __init__(self, res_n):
		super(AR_Net, self).__init__()

		nf = 64

		self.enc0 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)

		self.enc1 = nn.Sequential(
				nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2, bias=False),
				nn.LeakyReLU(0.2,inplace=True)
		)
		self.enc2 = nn.Sequential(
			nn.Conv2d(in_channels=128, out_channels=nf, kernel_size=3, stride=1, padding=1, bias=False),
			nn.LeakyReLU(0.2,inplace=True)
		)

		self.dec1 = nn.Sequential(
			nn.Conv2d(in_channels=nf, out_channels=128, kernel_size=3, stride=1, padding=1),
			nn.LeakyReLU(0.2,inplace=False),
			)
		self.dec2 = nn.Sequential(
			nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5, stride=1, padding=2),
			nn.LeakyReLU(0.2,inplace=False),
			)

		self.end_conv=nn.Conv2d(in_channels=64, out_channels=1, kernel_size=5, stride=1, padding=2, bias=False)

		# self.feature_enh = B.make_layer(B.Residual_Block_Spec(512), res_n)
		self.feature_enh = B.make_layer(B.RRDB(nf), res_n)

		self.enh_conv = nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, groups=1)

		self.final_act = nn.Tanh()

	def forward(self, x):

		x = self.enc0(x)
		x = self.enc1(x) #CONV LeakyReLU
		x = self.enc2(x) #CONV LeakyReLU

		identity = x
		x = self.feature_enh(x)
		#x = self.enh_conv(x)
		x = x + identity

		# x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
		x = self.dec1(x)  #CONV LeakyReLU

		# x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
		x = self.dec2(x) #CONV LeakyReLU

		x = self.end_conv(x)
		x = self.final_act(x)

		return x



class CbCr_Net(nn.Module):

	def __init__(self, res_n):
		super(CbCr_Net, self).__init__()

		nf = 64

		self.enc0_0 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(2,3,3), stride=1, padding=(0,1,1), bias=False)
		self.enc0_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)

		self.enc1 = nn.Sequential(
				nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2, bias=False),
				nn.LeakyReLU(0.2,inplace=True)
		)
		self.enc2 = nn.Sequential(
			nn.Conv2d(in_channels=128, out_channels=nf, kernel_size=3, stride=1, padding=1, bias=False),
			nn.LeakyReLU(0.2,inplace=True)
		)

		self.dec1 = nn.Sequential(
			nn.Conv2d(in_channels=nf, out_channels=128, kernel_size=3, stride=1, padding=1),
			nn.LeakyReLU(0.2,inplace=False),
			)
		self.dec2 = nn.Sequential(
			nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5, stride=1, padding=2),
			nn.LeakyReLU(0.2,inplace=False),
			)

		self.end_conv=nn.Conv2d(in_channels=64, out_channels=1, kernel_size=5, stride=1, padding=2, bias=False)

		# self.feature_enh = B.make_layer(B.Residual_Block_Spec(512), res_n)
		self.feature_enh = B.make_layer(B.RRDB(nf), res_n)

		self.enh_conv = nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=1, dilation=1, bias=True, groups=1)

		self.final_act = nn.Tanh()

	def forward(self, x):

		x = self.enc0_0(x.unsqueeze(1))
		x = self.enc0_1(x.squeeze(2))
		x = self.enc1(x) #CONV LeakyReLU
		x = self.enc2(x) #CONV LeakyReLU

		identity = x
		x = self.feature_enh(x)
		#x = self.enh_conv(x)
		x = x + identity

		# x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
		x = self.dec1(x)  #CONV LeakyReLU

		# x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
		x = self.dec2(x) #CONV LeakyReLU

		x = self.end_conv(x)
		x = self.final_act(x)

		return x

class CbCr_Net_double(nn.Module):

	def __init__(self, res_n):
		super(CbCr_Net_double, self).__init__()

		nf = 64

		self.enc0_0 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(3,3,3), stride=1, padding=(0,1,1), bias=False)
		self.enc0_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)

		self.enc1 = nn.Sequential(
				nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2, bias=False),
				nn.LeakyReLU(0.2,inplace=True)
		)
		self.enc2 = nn.Sequential(
			nn.Conv2d(in_channels=128, out_channels=nf, kernel_size=3, stride=1, padding=1, bias=False),
			nn.LeakyReLU(0.2,inplace=True)
		)

		self.dec1 = nn.Sequential(
			nn.Conv2d(in_channels=nf, out_channels=128, kernel_size=3, stride=1, padding=1),
			nn.LeakyReLU(0.2,inplace=False),
			)
		self.dec2 = nn.Sequential(
			nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5, stride=1, padding=2),
			nn.LeakyReLU(0.2,inplace=False),
			)

		self.end_conv=nn.Conv2d(in_channels=64, out_channels=2, kernel_size=5, stride=1, padding=2, bias=False)

		self.feature_enh = B.make_layer(B.RRDB(nf), res_n)

		self.final_act = nn.Tanh()

	def forward(self, x):

		x = self.enc0_0(x.unsqueeze(1))
		x = self.enc0_1(x.squeeze(2))
		x = self.enc1(x) #CONV LeakyReLU
		x = self.enc2(x) #CONV LeakyReLU

		identity = x
		x = self.feature_enh(x)
		#x = self.enh_conv(x)
		x = x + identity

		# x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
		x = self.dec1(x)  #CONV LeakyReLU

		# x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
		x = self.dec2(x) #CONV LeakyReLU

		x = self.end_conv(x)
		x = self.final_act(x)

		return x
