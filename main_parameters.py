#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 11:04:35 2022

@author: cmccurley
"""

"""
***********************************************************************
    *  File:  
    *  Name:  Connor H. McCurley
    *  Date:  
    *  Desc:  
    *
    *  Syntax: parameters = atr_parameters()
    *
    *  Outputs: parameters: dictionary holding parameters for atrMain.py
    *
    *
    *  Author: Connor H. McCurley
    *  University of Florida, Electrical and Computer Engineering
    *  Email Address: cmccurley@ufl.edu
    *  Latest Revision: 
    *  This product is Copyright (c) 2020 University of Florida
    *  All rights reserved
**********************************************************************
"""
    

import argparse

#######################################################################
########################## Define Parameters ##########################
#######################################################################
def set_parameters(args):
    """
    ******************************************************************
        *
        *  Func:    set_parameters()
        *
        *  Desc:    Class definition for the configuration object.  
        *           This object defines that parameters for the MIL U-Net
        *           script.
        *
    ******************************************************************
    """  
    
    ######################################################################
    ############################ Define Parser ###########################
    ######################################################################
    
    parser = argparse.ArgumentParser(description='Training script for bag-level classifier.')
    
    parser.add_argument('--run_mode', help='Mode to train, test, or compute CAMS. (cam)', default='test', type=str)
    
    ######################################################################
    ######################### Input Parameters ###########################
    ######################################################################
    
    parser.add_argument('--DATASET', help='Dataset selection.', default='mnist', type=str)
    parser.add_argument('--DATABASENAME', help='Relative path to the training data list.', default='', type=str)
 
    parser.add_argument('--target_classes', help='List of target classes.', nargs='+', default=['aeroplane'])
    parser.add_argument('--bg_classes', help='List of background classes.', nargs='+', default=['cat'])
    
#    parser.add_argument('--cams', help='List of CAMs to compute.', nargs='+', default=['gradcam','gradcam++','layercam','scorecam','ablationcam','eigencam'])
#    parser.add_argument('--CAM_SEG_THRESH', help='Hard threshold to convert CAM to segmentation decision.', nargs='+', default=[0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument('--layers', help='Layers to compute CAM at.', nargs='+', default=[4,9,16,23,30])
#    parser.add_argument('--CAM_SEG_THRESH', help='Hard threshold to convert CAM to segmentation decision.', nargs='+', default=[0.001, 0.01, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9])
    
    
    ## Feature ranking parameters
    parser.add_argument('--fitness_function', help='Fitness/scoring function for feature ranking. (miou or importance)', default='miou', type=str)
    parser.add_argument('--rank_features', help='Flag to train classification model.', default='False', type=eval,choices=[True,False])
    parser.add_argument('--create_training_set', help='Flag to train classification model.', default='False', type=eval, choices=[True,False])
    parser.add_argument('--train_test_classifier', help='Flag to train classification model.', default='False', type=eval, choices=[True,False])
    parser.add_argument('--feature_ranking_visualization', help='Flag to train classification model.', default='False', type=eval, choices=[True,False])
    parser.add_argument('--plot_top_activations', help='Flag to train classification model.', default='False', type=eval, choices=[True,False])
    parser.add_argument('--plot_ranked_features', help='Flag to train classification model.', default='False', type=eval, choices=[True,False])
    parser.add_argument('--miou_of_importance_ranks', help='Flag to train classification model.', default='False', type=eval, choices=[True,False])
    
    
    parser.add_argument('--NUM_SUBSAMPLE', help='Flag to train classification model.', default=200, type=int)
    parser.add_argument('--LARGEST_FEATURE_SET', help='Flag to train classification model.', default=200, type=int)
    
    
    ######################################################################
    ##################### Training Hyper-parameters ######################
    ######################################################################
    
    ## Starting parameters
#    parser.add_argument('--starting_parameters', help='Initial parameters when training from scratch. (0 uses pre-training on ImageNet.)', default='0', type=str)
    parser.add_argument('--starting_parameters', help='Initial parameters when training from scratch. (0 uses pre-training on ImageNet.)', default='/model_eoe_80.pth', type=str)
    parser.add_argument('--model', help='Neural network model', default='vgg16', type=str)
    
#    ## Hyperparameters
#    parser.add_argument('--NUM_CLASSES', help='Number of classes in the data set.', default=2, type=int)
    parser.add_argument('--BATCH_SIZE', help='Input batch size for training.', default=20, type=int)
    parser.add_argument('--EPOCHS', help='Number of epochs to train.', default=100, type=int)
    parser.add_argument('--LR', help='Learning rate.', default=0.001, type=float)
#    
#    ## Updating  
#    parser.add_argument('--update_on_epoch', help='Number of epochs to update loss script.', default=5, type=int)
    parser.add_argument('--parameter_path', help='Where to save/load weights after each epoch of training', default='/trainTemp.pth', type=str)
#    
    ## PyTorch/GPU parameters
    parser.add_argument('--no-cuda', action='store_true', help='Disables CUDA training.', default=False)
    parser.add_argument('--cuda', help='Enable CUDA training.', default=True)
    
    ######################################################################
    ######################### Output Parameters ##########################
    ######################################################################
    
    ## Output parameters
    parser.add_argument('--loss_file', help='Where to save training and validation loss updates.', default='/loss.txt', type=str)
    parser.add_argument('--outf', help='Where to save output files.', default='./output/mnist/vgg16/output', type=str)
    parser.add_argument('--NUMWORKERS', help='', default=0, type=int)
    
    
    parser.add_argument('--TMPDATAPATH', help='', default='', type=str)
    
    
    return parser.parse_args(args)
    
    
