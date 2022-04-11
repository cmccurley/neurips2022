# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 14:42:57 2019

@author: cmccurley
"""

"""
***********************************************************************
    *  File:  util.py
    *
    *  Desc: This file contains the helper and utility functions available 
    *        for MIL UNet.
    *
    *  Written by:  Guohao Yu, Weihuang Xu, Dr. Alina Zare, 
    *  Connor H. McCurley, Deanna L. Gran
    *
    *  Latest Revision:  2019-11-18
**********************************************************************
"""
######################################################################
######################### Import Packages ############################
######################################################################

## Default packages
import argparse
import numpy as np
from tqdm import tqdm

## Pytorch packages
import torch
import torch.nn as nn

######################################################################
##################### Function Definitions ###########################
######################################################################

        
def print_status(epoch, epoch_loss, train_loader, valid_loader, model, device, logfilename):
    """
    ******************************************************************
        *  Func:    print_status()
        *  Desc:    Updates the terminal with training progress and writes to loss text file.
        *  Inputs:  Epoch #, train loss, valid loss, desired output file, time taken to complete epoch
        *  Outputs: -----
    ******************************************************************
    """
    ############################# Training #############################
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    print('Computing training loss...')
    with torch.no_grad():
        for data in tqdm(train_loader):
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    training_accuracy = (100 * correct / total)
    
    ############################# Validation #############################
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    print('Computing validation loss...')
    with torch.no_grad():
        for data in tqdm(valid_loader):
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    validation_accuracy = (100 * correct / total)
    
    print('Training accuracy: %d %%' % training_accuracy)  
    print('Validation accuracy: %d %%' % validation_accuracy)    
    
    line0 = '=================================='
    line1 = 'epoch ' + str(epoch) + ', train loss=' + str(round(training_accuracy,2))
    line2 = 'epoch ' + str(epoch) + ', valid loss=' + str(round(validation_accuracy,2))
#    line4 = '=================================='
    f = open(logfilename, 'a+') 
    f.write(line0 + '\n')
    f.write(line1 + '\n') 
    f.write(line2 + '\n')   
#    f.write(line4 + '\n')
    f.close()
    
    return

def convert_gt_img_to_mask(gt_images, labels, parameters):
    
    try:
        gt_images = gt_images.detach().cpu().numpy()
    except:
        gt_images = gt_images.cpu().numpy()
        
    gt_image = gt_images[0,:,:]
                
    if(labels.detach().cpu().numpy()==0):
        for key in parameters.bg_classes:
            value = parameters.all_classes[key]
            gt_image[np.where(gt_image==value)] = 1
            
    elif(labels.detach().cpu().numpy()==1):
        for key in parameters.target_classes:
            value = parameters.all_classes[key]
            gt_image[np.where(gt_image==value)] = 1
    
    ## Convert segmentation image to mask
    gt_image[np.where(gt_image!=1)] = 0
    
    return gt_image

def cam_img_to_seg_img(cam_img, CAM_SEG_THRESH):
    
    cam_img[cam_img>=CAM_SEG_THRESH] = 1
    cam_img[cam_img<CAM_SEG_THRESH] = 0
    
    return cam_img




