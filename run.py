import argparse
import time

import numpy as np
import skimage.io as io
import torch
from models import AR_Net, CbCr_Net_double
from skimage.color import rgb2ycbcr, ycbcr2rgb
from torchvision.transforms import Compose, Normalize, ToTensor
from torchvision.utils import save_image

parser = argparse.ArgumentParser(description='Artifact Removal net')
parser.add_argument('-my', '--modely',  type=str, default='./weights/netAR.pth', help='Model to load')
parser.add_argument('-mc', '--modelc',  type=str, default='./weights/netCbCr.pth', help='Model to load')
parser.add_argument('-i', '--imgs',  nargs='*', type=str, default='', help='Input image')
parser.add_argument("--cuda", action="store_true", default=False , help="Use cuda?")

opt = parser.parse_args()

to_tensor = ToTensor()
norm = Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))

with torch.no_grad():

    ################################################################################
    ##################### General Options ##########################################

    cuda_check = torch.cuda.is_available() and opt.cuda

    ############################ Network model_ars ####################################
    print("\n===> Loading models")

    model_ar = AR_Net(5)

    model_cbcr = CbCr_Net_double(3)

    model_ar.load_state_dict(torch.load(opt.modely))
    model_cbcr.load_state_dict(torch.load(opt.modelc))

    ############################ Setting cuda ######################################
    print("\n===> Setting GPU")
    if cuda_check:
        model_ar.cuda()
        model_cbcr.cuda()

    for im_dir in opt.imgs:
        inputt = io.imread(im_dir)
        inputt = rgb2ycbcr(inputt).round().astype(np.float32)

        inputt = to_tensor(inputt)
        inputt = norm(inputt.div(255))
        name = im_dir
        name = name.split('/')[-1]
        name = name.split('.')[0]

        print('\n===> Image {}\n'.format(name))

        if cuda_check:
            inputt = inputt.cuda()

        y_input = inputt[0,:,:]
        y_input = y_input.unsqueeze(0).unsqueeze(0)
        cb_input = inputt[1,:,:]
        cb_input = cb_input.unsqueeze(0).unsqueeze(0)
        cr_input = inputt[2,:,:]
        cr_input = cr_input.unsqueeze(0).unsqueeze(0)

        print('\n===> Network started\n')
        st_time = time.time()

        y_clean = model_ar(y_input)

        cbcr_clean = model_cbcr(torch.cat((y_clean,cb_input, cr_input),1))

        result = torch.cat((y_clean,cbcr_clean),1)
        res_a = result[0].data.cpu().numpy()
        res_a = res_a.transpose(1,2,0)

        y_clean = (y_clean + 1)/2
        res_a = (res_a + 1)/2
        res_a = (res_a*255)
        rgb_result = ycbcr2rgb(res_a)
        rgb_result = rgb_result*255
        np.putmask(rgb_result, rgb_result > 255, 255)
        np.putmask(rgb_result, rgb_result < 0, 0)

        e_time = time.time() - st_time
        print('\n===> Image finished. Elapsed time: '+str(e_time))
        io.imsave('./outputs/'+name+'_clean.png', rgb_result.astype(np.uint8))
