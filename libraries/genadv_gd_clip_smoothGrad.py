#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 15:49:16 2019

@author: fcheng
"""

import os

import numpy as np


import torch as th
from torchvision import transforms
from torchvision import utils

from loss import esbAdvs_loss
from gaussian_blur import gaussian_blur



def gen_adv_image(initial_image, prob0, net_list, 
                      alpha = None, clip = 32,
                      layer_weight=None,
                      loss_type='l2',
                      iter_n=200,
                      lr_start=2., lr_end=1e-10,
                      momentum_start=0.9, momentum_end=0.9,
                      grad_normalize=True,
                      image_blur=True, kernel = (5, 5), sigma= (2, 2), 
                      disp_every=1,
                      save_intermediate=False, save_intermediate_every=1, save_intermediate_path=None
                      ):
    '''Generate adversarial images using gradient descent with momentum.

    Parameters
    ----------
    initial_image: ndarray
        Initial image for the optimization.
    prob0: ndarray
        The target category of the generated image.
    net_list: a list of caffe.Classifier or caffe.Net object
        All the artificial NNs to be attacked.

    Optional Parameters
    ---------- 
    alpha: ndarray
        The weights of each net for prob loss.
        Use equal weights for all nets by setting to None. 
    Lambda: single
        The parameter for balancing constraints of image loss and prob loss in the loss function.
        The larger the Lambda is, more important the image loss becomes.
    loss_type: str
        The loss function type: {'l2','l1','inner','gram'}.
    iter_n: int
        The total number of iterations.
    lr_start: float
        The learning rate at start of the optimization.
        The learning rate will linearly decrease from lr_start to lr_end during the optimization.
    lr_end: float
        The learning rate at end of the optimization.
        The learning rate will linearly decrease from lr_start to lr_end during the optimization.
    momentum_start: float
        The momentum (gradient descend with momentum) at start of the optimization.
        The momentum will linearly decrease from momentum_start to momentum_end during the optimization.
    momentum_end: float
        The momentum (gradient descend with momentum) at the end of the optimization.
        The momentum will linearly decrease from momentum_start to momentum_end during the optimization.
    grad_normalize: bool
        Normalise the gradient or not for each iteration.
    image_blur: bool
        Use image smoothing or not.
        If true, smoothing the image for each iteration.
    sigma_start: float
        The size of the gaussian filter for image smoothing at start of the optimization.
        The sigma will linearly decrease from sigma_start to sigma_end during the optimization.
    sigma_end: float
        The size of the gaussian filter for image smoothing at the end of the optimization.
        The sigma will linearly decrease from sigma_start to sigma_end during the optimization.
  
    p: float
        The order of the p-norm loss of image
    lamda_start: float
        The weight for p-norm loss at start of the optimization.
        The lamda will linearly decrease from lamda_start to lamda_end during the optimization.
    lamda_end: float
        The weight for p-norm loss at the end of the optimization.
        The lamda will linearly decrease from lamda_start to lamda_end during the optimization.
    disp_every: int
        Display the optimization information for every n iterations.
    save_intermediate: bool
        Save the intermediate reconstruction or not.
    save_intermediate_every: int
        Save the intermediate reconstruction for every n iterations.
    save_intermediate_path: str
        The path to save the intermediate reconstruction.

    Returns
    -------
    img: ndarray
        The reconstructed image [224x224x3].
    loss_list: ndarray
        The loss for each iteration.
        It is 1 dimensional array of the value of the loss for each iteration.
    '''
    
    dtype = th.cuda.FloatTensor
    
    # num_of_neural_net
    num_of_net = len(net_list)
    
    # iteration for gradient descent
    img = initial_image.copy()
    
    preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean =[0.485, 0.456, 0.406],
                                 std = [0.229, 0.224, 0.225])
            ])  
    
    deprocess = transforms.Compose([
 	    transforms.Normalize(mean = [-0.485/0.229, -0.456/0.224, -0.406/0.225],
				 std = [1./(0.229), 1./(0.224), 1./(0.225)])
    	    ])
   
    upper_bound = preprocess(img+clip).unsqueeze(0).type(dtype)
    lower_bound = preprocess(img-clip).unsqueeze(0).type(dtype)
    img = preprocess(img).unsqueeze(0)
    delta_img = th.zeros((1,3,224,224)).type(dtype)
    sav_img = th.zeros((1,3,224,224)).type(dtype)
    loss_list = np.zeros(iter_n, dtype='float32')
    loss_list[:] = np.nan
    for t in range(iter_n):
        
        
        # parameter
        lr = lr_start + t * (lr_end - lr_start) / iter_n
        momentum = momentum_start + t * (momentum_end - momentum_start) / iter_n    

        img = img.type(dtype)
        img.requires_grad = True
        
        # forward
        prob = th.zeros((num_of_net, 1000)).type(dtype)
        for j in range(num_of_net):
            
            net_j = net_list[j]
            output_j = net_j(img)
            prob_j = th.exp(output_j)/th.sum(th.exp(output_j))
            prob[j] = prob_j
            
        prob_loss, obj_grad = esbAdvs_loss(prob, prob0, alpha)
        loss_list[t] = prob_loss.item()
        loss_min = np.nanmin(loss_list)
        if loss_list[t] == loss_min:
            sav_img = img

        # backward for net
        # net_grad = th.zeros((num_of_net, 3, h, w))
        if img.grad is not None:
            img.grad.data.zero_()
        
        for j in range(num_of_net):
            
            # img.grad.data.zero_()
            net_j = net_list[j]
            net_j.zero_grad()
            
            prob_j = prob[j]
            prob_j.backward(obj_grad.squeeze(), retain_graph=True)
            
        if np.isnan(loss_list[t]):
            grad = th.randn_like(img).type(dtype)
            momentum = 0
        else:    
            grad = img.grad.data/num_of_net

        # normalize gradient
        if grad_normalize:
            grad_mean = th.mean(th.abs(grad))
            if grad_mean > 0:
                grad = grad / grad_mean

        if image_blur:
            grad = gaussian_blur(grad, kernel, sigma)
        
        # gradient with momentum
        delta_img = delta_img * momentum + grad

        # image update
        img = img - lr * delta_img
      
        # clip image so that it's in the range of [img0-clip,img0+clip]
        img = img.detach()
        img = th.max(th.min(img, upper_bound), lower_bound)

        # disp info
        if (t+1) % disp_every == 0:
            print('iter=%d; err=%g' % (t+1, prob_loss))

        # save image
        if save_intermediate and ((t+1) % save_intermediate_every == 0):
            save_name = '%05d.jpg' % (t+1)
            utils.save_image(deprocess(img.squeeze()), os.path.join(save_intermediate_path, save_name))
            
    sav_img = deprocess(sav_img.squeeze())

    # return img
    return sav_img, loss_list
