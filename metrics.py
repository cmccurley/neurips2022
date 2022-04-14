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


"""
%=====================================================================
%===================== Faithfulness Functions ========================
%=====================================================================
"""


######################################################################
##################### Faithfulness Functions #########################
######################################################################
## Compute ID (Higher value is better)
def compute_ID(input_image, cam_heatmap, image_label, model):
    
    ## Compute explanation image
    cam_image = np.matmul(cam_heatmap, input_image)
    
    ## Pull out softmax outputs for class (image_label)
    Y = model(input_image)
    O = model(cam_image)
    
    ## Compare if cam image increases class prediction
    if (O > Y):
        ID = 1
    else:
        ID = 0
    
    return ID

## Compute AD (Lower value is better)
def compute_AD(input_image, cam_heatmap, image_label, model):
    
    ## Compute explanation image
    cam_image = np.matmul(cam_heatmap, input_image)
    
    ## Pull out softmax outputs for class (image_label)
    Y = model(input_image)
    O = model(cam_image)
    
    ## Compute Average Drop
    AD = max(0,Y - O)/Y
    
    return AD


## Compute ADD (Higher is better)
def compute_ADD(input_image, cam_heatmap, image_label, model):
    
    ## Compute inverted explanation image
    cam_image = np.matmul((1 - cam_heatmap), input_image)
    
    ## Pull out softmax outputs for class (image_label)
    Y = model(input_image)
    D = model(cam_image)
    
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