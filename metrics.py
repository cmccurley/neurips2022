#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 15:25:46 2022

@author: cmccurley
"""

"""
%=====================================================================
%================= Import Settings and Packages ======================
%=====================================================================
"""
import numpy as np
import torch
import torch.nn as nn


"""
%=====================================================================
%===================== Faithfulness Functions ========================
%=====================================================================
"""

"""
%=====================================================================
Metrics as defined by H. Jung and Y. Oh, 
"Towards Better Explanations of Class Activation Mapping," 
in CoRR abs/2102.05228, 2021.
%=====================================================================
"""

######################################################################
##################### Faithfulness Functions #########################
######################################################################

## Compute ID (Higher value is better)
def compute_ID(input_image, cam_heatmap, image_label, model, device):
    
    ##
    ## Normalized input image, cam_heatmap computed on predicted label, true image class, model
    ##
    
    ## Compute explanation image
    image_permuted = input_image.permute(2,3,1,0)

    ## Compute explanation image
    mask_3_channel_img = np.repeat(np.expand_dims(cam_heatmap,axis=2),3,axis=2) ## Repeat mask for hadamard product
    mask_4_channel_img = torch.Tensor(np.expand_dims(mask_3_channel_img,axis=3)).to(device)
    explanation_img = torch.mul(mask_4_channel_img,image_permuted)
    explanation_image = explanation_img.permute(3,2,0,1) ## Reshape into tensor for model
    
    ## Pull out softmax outputs for class (image_label)
    softmax = nn.Softmax(dim=-1)
    Y = softmax(model(input_image))[0,image_label]
    O = softmax(model(explanation_image))[0,image_label]
    
    ## Compare if cam image increases class prediction
    if (O > Y):
        ID = 1
    else:
        ID = 0
    
    return ID

## Compute AD (Lower value is better)
def compute_AD(input_image, cam_heatmap, image_label, model, device):
    ##
    ## Normalized input image, cam_heatmap computed on predicted label, true image class, model
    ##
    
    ## Compute explanation image
    image_permuted = input_image.permute(2,3,1,0)

    ## Compute explanation image
    mask_3_channel_img = np.repeat(np.expand_dims(cam_heatmap,axis=2),3,axis=2) ## Repeat mask for hadamard product
    mask_4_channel_img = torch.Tensor(np.expand_dims(mask_3_channel_img,axis=3)).to(device)
    explanation_img = torch.mul(mask_4_channel_img,image_permuted)
    explanation_image = explanation_img.permute(3,2,0,1) ## Reshape into tensor for model
    
    ## Pull out softmax outputs for class (image_label)
    softmax = nn.Softmax(dim=-1)
    Y = softmax(model(input_image))[0,image_label]
    O = softmax(model(explanation_image))[0,image_label]
    
    ## Compute Average Drop
    AD = max(0,Y - O)/Y
    
    return AD


## Compute ADD (Higher is better)
def compute_ADD(input_image, cam_heatmap, image_label, model, device):
    ##
    ## Normalized input image, cam_heatmap computed on predicted label, true image class, model
    ##
    
    ## Compute explanation image
    image_permuted = input_image.permute(2,3,1,0)

    ## Compute INVERTED explanation image
    mask_3_channel_img = np.repeat(np.expand_dims(np.abs(1 - cam_heatmap),axis=2),3,axis=2) ## Repeat mask for hadamard product
    mask_4_channel_img = torch.Tensor(np.expand_dims(mask_3_channel_img,axis=3)).to(device)
    inv_explanation_img = torch.mul(mask_4_channel_img,image_permuted)
    inv_explanation_image = inv_explanation_img.permute(3,2,0,1) ## Reshape into tensor for model
    
    ## Pull out softmax outputs for class (image_label)
    softmax = nn.Softmax(dim=-1)
    Y = softmax(model(input_image))[0,image_label]
    D = softmax(model(inv_explanation_image))[0,image_label]
    
    ## Compute Average Drop in Deletion
    ADD = (Y - D)/Y
    
    return ADD


"""
%=====================================================================
%===================== Localization Functions ========================
%=====================================================================
"""
######################################################################
##################### Localization Functions #########################
######################################################################

## Compute energy-based pointing game
def compute_EBG():
    
    return EBG


"""
%=====================================================================
%===================== Segmentation Functions ========================
%=====================================================================
"""
######################################################################
##################### Segmentation Functions #########################
######################################################################

## Compute IoU pointing game
def compute_IOU():
    
    return IOU