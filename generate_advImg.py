"""
Created on Wed Jun 26 13:49:22 2019

@author: fcheng  chengfanbrain@gmail.com
"""


import argparse
import datetime
import os
import sys

from PIL import Image
import numpy as np
import fnmatch

import torch as th
import torchvision as thv
from torchvision import utils

sys.path.append('./libraries/')
from utils import normalise_img
from genadv_gd_clip_smoothGrad import gen_adv_image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate adversarial images that fool multiple DNNs')
    parser.add_argument('--target_category', type=str, help='target category (1-1000)', default=289)
    parser.add_argument('--purturbation_level', type=int, help='maximum pixel intensity changei (default: 16)', default=16)
    parser.add_argument('--image_blur', type=bool, help='whether to smooth images (default: True)', default=True)
    parser.add_argument('--loss_type', type=str, help='optimization loss type (l2, l1, inner, gram)', default='l2')
    parser.add_argument('--iter_num', type=int, help='number of iterations for gradient descend (default: 100)', default=100)
    parser.add_argument('--lr_start', type=float, help='initial learning rate (default: 3)', default=3)
    parser.add_argument('--lr_end', type=float, help='final learning rate (default: 1e-10)', default=1e-10)
    parser.add_argument('--momentum_start', type=float, help='initial momentum (default: 0.9)', default=0.9)
    parser.add_argument('--momentum_end', type=float, help='final momentum (default: 0.9)', default=0.9) 
    parser.add_argument('--file_extension', type=str, help='file extension for images', default='*.jpg')
    parser.add_argument('--root_dir', type=str, help='root_directory', default='./') 
    parser.add_argument('--img_dir', type=str, help='location for saving original images', default='./image')
    parser.add_argument('--save_intermediate', type=bool, help='whether to save the intermediate images (default: True)', default=True)
    parser.add_argument('--save_intermediate_every', type=int, help='save the intermediate images for every n iterations', default=20)
    parser.add_argument('--device_id', type=int, nargs='+', help='list of CUDA devices (default: 0)', default=0)
    args = parser.parse_args()

    device = f"cuda:{args.device_id}" if th.cuda.is_available() else "cpu"
    th.cuda.set_device(args.device_id)
    print(device)
   
    dtype = th.cuda.FloatTensor
    # purturbation level
    clip = args.purturbation_level
   
    # initial setting
    save_dir = args.root_dir


    # Load original image 
    img_dir = args.img_dir
    fn_pattern = args.file_extension
    img_fn_list = []
    for root, _, fn_list in os.walk(img_dir):
       for fn in sorted(fnmatch.filter(fn_list, fn_pattern)):
           img_fn_list.append(os.path.join(root, fn))


    # Load CNN model for training
    h, w = 224, 224
    alexnet = thv.models.alexnet(pretrained = True)
    vgg16 = thv.models.vgg16(pretrained = True)
    vgg19 = thv.models.vgg19(pretrained = True)
    vgg16_bn = thv.models.vgg16_bn(pretrained = True)
    vgg19_bn = thv.models.vgg19_bn(pretrained = True)
    densenet161 = thv.models.densenet161(pretrained = True)
    densenet201 = thv.models.densenet201(pretrained = True)
    inception_v3 = thv.models.inception_v3(pretrained = True)
    resnet50 = thv.models.resnet50(pretrained = True)
    resnet101 = thv.models.resnet101(pretrained = True)
    resnet152 = thv.models.resnet152(pretrained = True)
    squeezenet1_1 = thv.models.squeezenet1_1(pretrained = True)
    googlenet= thv.models.googlenet(pretrained = True)
    mobilenet_v2 = thv.models.mobilenet_v2(pretrained = True)
    resnext101_32x8d = thv.models.resnext101_32x8d(pretrained = True)

    net_list = [alexnet, vgg16, vgg19, vgg16_bn, vgg19_bn,
               densenet161, densenet201, inception_v3, 
               resnet50, resnet101, resnet152, squeezenet1_1, googlenet, 
               mobilenet_v2, resnext101_32x8d];

    for net in net_list:
       net = net.to(device)
       net.eval()
    
    # parameter setting, each CNN shares the same weight for the loss term
    num_of_net = len(net_list)
    alpha = (th.ones((1, num_of_net))/num_of_net).type(dtype)
    
    #the image index list
    img_idx_list = [2]
    img_labs = [285]
    # target category index in 1-1000
    tgt_lab = args.target_category
    
    # Make directory for saving the results
    save_subdir = 'train' + str(num_of_net)+'_withGaussianFilter_clip' + str(clip) +'_' + datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    save_path = os.path.join(save_dir, save_subdir)
    os.makedirs(save_path)
    
    for idx in range(len(img_idx_list)):
        
            
        i = img_idx_list[idx]
        img_lab = img_labs[idx]
        # save directory
        img_path = os.path.join(save_path, str(i+1))
        os.makedirs(img_path)
            
        # original image or initial image
        initial_image = Image.open(img_fn_list[i])
        # save initial image
        initial_image = initial_image.resize((h, w))
        initial_image = normalise_img(np.asarray(initial_image), clip)
        save_name = 'Img_orig_' + format(img_lab, '04d') +'.jpg'
        Image.fromarray(initial_image).save(os.path.join(img_path, save_name))
        
        # The targeted label
 
        prob0 =  th.zeros((1,1000)).type(dtype)
        prob0[0,tgt_lab-1] = 1
            
            
        # Iteration --------------------------------------------------------------
        # options
        opts = {
               # alpha: ndarray, the weights of each net for prob loss.
               'alpha': alpha,                
               # clip: purturbation level 
               'clip': clip,
               # Loss function type: {'l2', 'l1', 'inner', 'gram'}
               'loss_type': args.loss_type,
            
               # The total number of iterations for gradient descend
               'iter_n': args.iter_num,
               # Display the information for every n iterations
  	       'disp_every': 1,
               # Save the intermediate reconstruction or not
               'save_intermediate': args.save_intermediate,
               # Save the intermediate reconstruction for every n iterations
               'save_intermediate_every': args.save_intermediate_every,
               # Path to the directory saving the intermediate reconstruction
               'save_intermediate_path': img_path,
            
               # Learning rate
               'lr_start': args.lr_start,
               'lr_end': args.lr_end,
               # Gradient with momentum
               'momentum_start': args.momentum_start,
               'momentum_end': args.momentum_end,                    
               # normalize gradient
               'grad_normalize': True,
                    
               # Use image smoothing or not
               'image_blur': args.image_blur,
               # The size of the gaussian filter for image smoothing
               'kernel': (5, 5),
               'sigma': (2, 2),
               }
          
        # generate adversarial images
        adv_img, loss_list = gen_adv_image(initial_image, prob0, net_list,  **opts)
                    
        # Save the results ------------------------------------------------------------
        save_name = 'Img_orig_' + format(img_lab, '04d') + '_tgt_' + format(tgt_lab, '04d') + '.jpg'
        utils.save_image(adv_img, os.path.join(img_path, save_name))
            
        
