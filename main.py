#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 16:14:46 2022

@author: cmccurley
"""


"""
***********************************************************************
    *  File:  
    *  Name:  Connor H. McCurley
    *  Date:  
    *  Desc:  
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


"""
%=====================================================================
%================= Import Settings and Packages ======================
%=====================================================================
"""

######################################################################
######################### Import Packages ############################
######################################################################

## General packages
import os
import json
import argparse
from tqdm import tqdm
import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn.metrics import precision_recall_fscore_support as prfs
from skimage.filters import threshold_otsu
from sklearn.linear_model import LogisticRegression

# PyTorch packages
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split

## Custom packages
import feature_ranking_parameters
import initialize_network
from util import convert_gt_img_to_mask, cam_img_to_seg_img
from dataloaders import loaderPASCAL, loaderDSIAC

import numpy as np
import matplotlib.pyplot as plt
from cam_functions.utils.image import show_cam_on_image, preprocess_image
from cam_functions import GradCAM, LayerCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, EigenCAM
from cam_functions import ActivationCAM

torch.manual_seed(24)
torch.set_num_threads(1)


"""
%=====================================================================
%============================ Main ===================================
%=====================================================================
"""

if __name__== "__main__":
    
    print('================= Running Main =================\n')
    
    ######################################################################
    ######################### Set Parameters #############################
    ######################################################################
    
    ## Import parameters 
    args = None
    parameters = feature_ranking_parameters.set_parameters(args)
    
    ## Define data transforms
    if (parameters.DATASET == 'dsiac'):
        transform = transforms.Compose([transforms.ToTensor()])
    else:
        transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Resize(256), 
                                    transforms.CenterCrop(224)])         

    target_transform = transforms.Compose([transforms.ToTensor()])
    
    ## Define dataset-specific parameters
    
    if (parameters.DATASET == 'cifar10'):
    
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 
                   'horse', 'ship', 'truck')
        
        parameters.DATABASENAME = '../../Data/cifar10'
        
        ## Create data objects from training, validation and test .txt files set in the parameters file
        trainset = torchvision.datasets.CIFAR10(root=parameters.DATABASENAME, 
                                                train=True, 
                                                download=False, 
                                                transform=transform)
        val_size = 5000
        train_size = len(trainset) - val_size
        train_ds, val_ds = random_split(trainset, [train_size, val_size])
        
        testset = torchvision.datasets.CIFAR10(root=parameters.DATABASENAME, 
                                               train=False, 
                                               download=False, 
                                               transform=transform)
        
        ## Create data loaders for the data subsets.  (Handles data loading and batching.)
        train_loader = torch.utils.data.DataLoader(train_ds, 
                                                   batch_size=parameters.BATCH_SIZE, 
                                                   shuffle=True, 
                                                   num_workers=0, 
                                                   collate_fn=None, 
                                                   pin_memory=False)
        valid_loader = torch.utils.data.DataLoader(val_ds, 
                                                 batch_size=parameters.BATCH_SIZE*2, 
                                                 shuffle=True, 
                                                 num_workers=0, 
                                                 collate_fn=None, 
                                                 pin_memory=False)
        test_loader = torch.utils.data.DataLoader(testset, 
                                                  batch_size=parameters.BATCH_SIZE, 
                                                  shuffle=False, 
                                                  num_workers=0, 
                                                  collate_fn=None, 
                                                  pin_memory=False)
        
    elif (parameters.DATASET == 'pascal'):
        
        parameters.pascal_data_path = '/home/UFAD/cmccurley/Public_Release/Army/Data/pascal/VOCdevkit/VOC2012/ImageSets/Main'
        parameters.pascal_image_path = '/home/UFAD/cmccurley/Public_Release/Army/Data/pascal/VOCdevkit/VOC2012/JPEGImages'
        parameters.pascal_gt_path = '/home/UFAD/cmccurley/Public_Release/Army/Data/pascal/VOCdevkit/VOC2012/SegmentationClass'
        
        parameters.all_classes = {'aeroplane':1, 'bicycle':2, 'bird':3, 'boat':4,
        'bottle':5, 'bus':6, 'car':7, 'cat':8, 'chair':9,
        'cow':10, 'diningtable':11, 'dog':12, 'horse':13,
        'motorbike':14, 'person':15, 'pottedplant':16,
        'sheep':17, 'sofa':18, 'train':19, 'tvmonitor':20}
        
        classes = parameters.bg_classes + parameters.target_classes
        
        parameters.NUM_CLASSES = len(classes)
        
        parameters.pascal_im_size = (224,224)
        
        trainset = loaderPASCAL(parameters.target_classes, parameters.bg_classes, 'train', parameters, transform, target_transform)
        validset = loaderPASCAL(parameters.target_classes, parameters.bg_classes, 'val', parameters, transform, target_transform)  
        testset = loaderPASCAL(parameters.target_classes, parameters.bg_classes, 'test', parameters, transform, target_transform) 
        
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=parameters.BATCH_SIZE, shuffle=True, pin_memory=False)
        valid_loader = torch.utils.data.DataLoader(validset, batch_size=parameters.BATCH_SIZE, shuffle=True, pin_memory=False) 
        test_loader = torch.utils.data.DataLoader(testset, batch_size=parameters.BATCH_SIZE, shuffle=True, pin_memory=False)
        
    elif (parameters.DATASET == 'dsiac'):
        
        parameters.dsiac_data_path = '/home/UFAD/cmccurley/Public_Release/Army/Data/DSIAC/bicubic'
        parameters.dsiac_gt_path = '/home/UFAD/cmccurley/Data/Army/Data/ATR_Extracted_Data/ground_truth_labels/dsiac'
        
        parameters.bg_classes = ['background']
        parameters.target_classes = ['target']
        parameters.all_classes = {'background':0, 'target':1}
        
        classes = parameters.bg_classes + parameters.target_classes
        parameters.NUM_CLASSES = len(classes)
        parameters.dsiac_im_size = (510,720)
        
        trainset = loaderDSIAC('../input/full_path/train_dsiac.txt', parameters, 'train', transform, target_transform) 
        validset = loaderDSIAC('../input/full_path/valid_dsiac.txt', parameters, 'valid', transform, target_transform) 
        testset = loaderDSIAC('../input/full_path/test_dsiac.txt', parameters, 'test', transform, target_transform) 
        
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=parameters.BATCH_SIZE, shuffle=True, pin_memory=False)
        valid_loader = torch.utils.data.DataLoader(validset, batch_size=parameters.BATCH_SIZE, shuffle=True, pin_memory=False) 
        test_loader = torch.utils.data.DataLoader(testset, batch_size=parameters.BATCH_SIZE, shuffle=True, pin_memory=False)
            

    ## Define files to save epoch training/validation
    logfilename = parameters.outf + parameters.loss_file

    ######################################################################
    ####################### Initialize Network ###########################
    ######################################################################
    
    model = initialize_network.init(parameters)
    
    ## Save initial weights for further training
    temptrain = parameters.outf + parameters.parameter_path
    torch.save(model.state_dict(), temptrain)
    
    ## Detect if GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
                    
    ###############################################################################
    ######################## Perform Feature Ranking ##############################
    ###############################################################################
    
    norm_image = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    ## Turn on gradients for CAM computation 
    for param in model.parameters():
        param.requires_grad = True

    model.eval()             
            
    activation_models = dict()
    
    if (parameters.model == 'vgg16'):
        for layer_idx in parameters.layers:
            activation_models[str(layer_idx)] = ActivationCAM(model=model, target_layers=[model.features[layer_idx]], use_cuda=parameters.cuda)
            
    groundtruth_model = LayerCAM(model=model, target_layers=[model.features[23]], use_cuda=parameters.cuda) ## LayerCAM stage 4

    target_category = None
    
    experiment_savepath = parameters.outf.replace('output','feature_ranking')
         
    ###########################################################################
    ########################### Extract Training Set ##########################
    ###########################################################################
    
    num_pos_samples = 0
    
    if parameters.rank_features:
        print('\nPerforming feature ranking...\n')
        
        for idx, data in enumerate(tqdm(train_loader)):
            
            ## Load sample and groundtruth
            images, labels, gt_images = data[0].to(device), data[1].to(device), data[2]
            
            ###############################################################
            ################### Get Pseudo-groundtruth ####################
            ###############################################################
            
    #         ## Get predicted class label
    #        output = model(cam_input)
    #        pred_label = np.argmax(output.detach().cpu().numpy()[0,:])
            pred_label = int(labels.detach().cpu().numpy())
            
            if pred_label:
            
                gt_input = norm_image(images)
                
                gt_img = groundtruth_model(input_tensor=gt_input, target_category=int(pred_label))
                gt_img = gt_img[0, :]
                
                gt_thresh = threshold_otsu(gt_img)
                gt_img = gt_img > gt_thresh
                
                
#                ## Visualize input and CAMs
#                images = images.permute(2,3,1,0)
#                images = images.detach().cpu().numpy()
#                image = images[:,:,:,0]
#                img = image
#                
#                cam_image = show_cam_on_image(img, gt_img, True)
#                plt.figure()
#                plt.imshow(cam_image)
#                plt.axis('off')
                
                ###############################################################
                ################# Extract activation maps #####################
                ###############################################################
                
                ## Normalize input to the model
                cam_input = norm_image(images)
                
                for stage_idx, activation_model_idx in enumerate(activation_models):
                
                    ## Get activation maps
                    activations, importance_weights = activation_models[activation_model_idx](input_tensor=cam_input, target_category=int(pred_label))
                    importance_weights = importance_weights[0,:]
                    
                    if not(stage_idx):
                        all_activations = activations
                        all_importance_weights = importance_weights
                    else:
                        all_activations = np.append(all_activations, activations, axis=0)
                        all_importance_weights = np.append(all_importance_weights, importance_weights, axis=0)
                    
                ###################################################################
                ###################### Evaluate Fitness ###########################
                ###################################################################
                
                ## Initialize matrix [num_samples, num_features] to hold fitness values
                num_features = all_activations.shape[0]
        
                if (parameters.fitness_function == 'miou'):
        
                    ## Evaluate fitness for each feature
                    sample_fitness = np.zeros(num_features)
                    for idk in range(num_features):
                        
                        ## Binarize feature
                        try:
                            img_thresh = threshold_otsu(all_activations[idk,:,:])
                            binary_feature_map = all_activations[idk,:,:] > img_thresh
                        except:
                            binary_feature_map = all_activations[idk,:,:] < 0.1
            
                        ## Compute fitness as IoU to pseudo-groundtruth
                        intersection = np.logical_and(binary_feature_map, gt_img)
                        union = np.logical_or(binary_feature_map, gt_img)
                        
                        ## Catch divide by zero(union of prediction and groundtruth is empty)
                        try:
                            iou_score = np.sum(intersection) / np.sum(union)
                        except:
                            iou_score = 0
                            
                        sample_fitness[idk] = iou_score
                        
                elif (parameters.fitness_function == 'importance'):
                    sample_fitness = all_importance_weights    
                
                ## Add values to global matrix
                sample_fitness = np.expand_dims(sample_fitness,axis=0)
                if not(num_pos_samples):
                    fitness_values = sample_fitness
                else:
                    fitness_values = np.concatenate((fitness_values, sample_fitness), axis=0)
                
#                fitness_values[num_pos_samples,:] = sample_fitness
        
                num_pos_samples += 1
                
        ###################################################################
        ######################### Rank Features ###########################
        ###################################################################
        
        ## Rank features
        avg_fitness_values = np.mean(fitness_values, axis = 0)
        
        sorted_idx = (-avg_fitness_values).argsort() ## indices of features in descending order
        sorted_fitness_values = avg_fitness_values[sorted_idx]
        
        ## Save learned feature ranking
        if (parameters.fitness_function == 'miou'):
            idx_path = parameters.outf.replace('output','feature_ranking') + '/sorted_feature_idx_miou'
            fitness_path = parameters.outf.replace('output','feature_ranking') + '/sorted_feature_fitness_miou'
        elif (parameters.fitness_function == 'importance'):
            idx_path = parameters.outf.replace('output','feature_ranking') + '/sorted_feature_idx_importance'
            fitness_path = parameters.outf.replace('output','feature_ranking') + '/sorted_feature_fitness_miou_importance'
        
        np.save(idx_path,sorted_idx)
        np.save(fitness_path,sorted_fitness_values)
         
    ###########################################################################
    ######################### Create Training Data Set ########################
    ###########################################################################
    
    if parameters.create_training_set:
        print('\nLoading training data...\n')
        
        ## Load ranked feature order
        if (parameters.fitness_function == 'miou'):
                idx_path = parameters.outf.replace('output','feature_ranking') + '/sorted_feature_idx_miou.npy'
                fitness_path = parameters.outf.replace('output','feature_ranking') + '/sorted_feature_fitness_miou.npy'
        elif (parameters.fitness_function == 'importance'):
            idx_path = parameters.outf.replace('output','feature_ranking') + '/sorted_feature_idx_importance.npy'
            fitness_path = parameters.outf.replace('output','feature_ranking') + '/sorted_feature_fitness_miou_importance.npy'
        
        sorted_idx = np.load(idx_path, allow_pickle=True)
        sorted_fitness_values = np.load(fitness_path, allow_pickle=True)
        
        ###############################################################
        #################### Extract training set #####################
        ###############################################################
        num_pos_samples = 0
        
        for sample_idx, data in enumerate(tqdm(train_loader)):
            
            ## Load sample and groundtruth
            images, labels, gt_images = data[0].to(device), data[1].to(device), data[2]
            
            pred_label = int(labels.detach().cpu().numpy())
            
            if pred_label:
            
                ###############################################################
                ####################### Get Groundtruth #######################
                ###############################################################
                
                ## Get pseudo-groundtruth from CAM
                gt_input = norm_image(images)
                
                gt_img = groundtruth_model(input_tensor=gt_input, target_category=int(pred_label))
                gt_img = gt_img[0, :]
                
                gt_thresh = threshold_otsu(gt_img)
                gt_img[gt_img>gt_thresh] = 1
                gt_img[gt_img<=gt_thresh] = 0

                labels = np.reshape(gt_img,(gt_img.shape[0]*gt_img.shape[1]))
                
                ###############################################################
                ################# Extract activation maps #####################
                ###############################################################
                
                ## Normalize input to the model
                cam_input = norm_image(images)
                
                for stage_idx, activation_model_idx in enumerate(activation_models):
                
                    ## Get activation maps
                    activations, importance_weights = activation_models[activation_model_idx](input_tensor=cam_input, target_category=int(pred_label))
                    importance_weights = importance_weights[0,:]
                    
                    if not(stage_idx):
                        all_activations = activations
                    else:
                        all_activations = np.append(all_activations, activations, axis=0)
    
    
                data = np.zeros((all_activations.shape[0],
                                 all_activations.shape[1]*all_activations.shape[2]))
        
                for idx in range(all_activations.shape[0]):
                    data[idx,:] = np.reshape(all_activations[idx,:,:],
                        (1,all_activations[idx,:,:].shape[0]*all_activations[idx,:,:].shape[1]))
                
                ###################################################################
                ###################### Sub-sample Data ############################
                ###################################################################
                num_samples = data.shape[1]
                
                subsampled_idx = np.arange(0,num_samples,3)
                
#                subsampled_idx = np.random.permutation(num_samples)[0:parameters.NUM_SUBSAMPLE]
                data_sampled = data[:,subsampled_idx]
                labels_sampled = labels[subsampled_idx]
                
                ###################################################################
                ############# Add to all training data and groundtruth ############
                ###################################################################
                
                if not(sample_idx):
                    train_data = data_sampled
                    train_labels = labels_sampled
                else:
                    train_data = np.append(train_data, data_sampled, axis=1)
                    train_labels = np.append(train_labels, labels_sampled)

                num_pos_samples += 1
                
                if (num_pos_samples > 5):
                    break
            
        ## Save training data set
        training_data_path = parameters.outf.replace('output','feature_ranking') + '/training_data'
        training_labels_path = parameters.outf.replace('output','feature_ranking') + '/training_labels'
        
        np.save(training_data_path,train_data)
        np.save(training_labels_path,train_labels)
        
        
    ###########################################################################
    ################################ Test Model ###############################
    ###########################################################################
    if parameters.train_test_classifier:
        
        ## Load training data
        training_data_path = parameters.outf.replace('output','feature_ranking') + '/training_data.npy'
        training_labels_path = parameters.outf.replace('output','feature_ranking') + '/training_labels.npy'
            
        train_data = np.load(training_data_path, allow_pickle = True)
        train_labels = np.load(training_labels_path, allow_pickle = True)
        
        ## Load ranked feature order
        if (parameters.fitness_function == 'miou'):
                idx_path = parameters.outf.replace('output','feature_ranking') + '/sorted_feature_idx_miou.npy'
                fitness_path = parameters.outf.replace('output','feature_ranking') + '/sorted_feature_fitness_miou.npy'
        elif (parameters.fitness_function == 'importance'):
            idx_path = parameters.outf.replace('output','feature_ranking') + '/sorted_feature_idx_importance.npy'
            fitness_path = parameters.outf.replace('output','feature_ranking') + '/sorted_feature_fitness_miou_importance.npy'
        
        sorted_idx = np.load(idx_path, allow_pickle=True)
        sorted_fitness_values = np.load(fitness_path, allow_pickle=True)
        
        ## Balance training set
        num_neg = len(train_labels[train_labels==0])
        num_pos = len(train_labels[train_labels==1])
        
        idx_neg = np.where(train_labels==0)[0]
        idx_pos = np.where(train_labels==1)[0]
        
        idx_neg_new = np.random.permutation(idx_neg)
        idx_neg_new = idx_neg_new[0:num_pos]
        
        ## Combine all indices
        data_indices = np.concatenate((idx_neg_new,idx_pos))
        data_indices = np.random.permutation(data_indices)
        
        training_data = train_data[:,data_indices]
        training_labels = train_labels[data_indices]
        
        ################################################
        data = training_data[:,0:4000]
        labels = training_labels[0:4000]
        del training_data
        del training_labels
        
        training_data = data
        training_labels = labels
        
        del data
        del labels
        ################################################
        
        
        print('\nTraining/Testing model...\n')
        
        results_dict = dict()
        
        for feature_set_index, feature_val in enumerate(sorted_idx[0:parameters.LARGEST_FEATURE_SET]):
        
            if not(feature_set_index):
                feature_set = [feature_val]
            else:
                feature_set.append(feature_val)
                
            print(f'{len(feature_set)}')
        
            ## define classifier
            clf = LogisticRegression(random_state=24,solver='sag').fit(training_data[feature_set,:].T,training_labels)
        
        
            n_samples = len(test_loader)
            metric_iou = np.zeros(n_samples, dtype="float32")
        
            num_pos_sample = 0
        
            for sample_idx, data in enumerate(tqdm(test_loader)):
          
                images, labels, gt_images = data[0].to(device), data[1].to(device), data[2]
                
                ## Normalize input to the model
                cam_input = norm_image(images)
                
                ## Get predicted class label
                output = model(cam_input)
                pred_label = np.argmax(output.detach().cpu().numpy()[0,:])
                
#                pred_label = int(labels.detach().cpu().numpy())
                    
                ############# Convert groundtruth image into mask #############
                gt_img = convert_gt_img_to_mask(gt_images, labels, parameters)
                
                ###############################################################
                ################# Extract activation maps #####################
                ###############################################################
                
                for stage_idx, activation_model_idx in enumerate(activation_models):
                
                    ## Get activation maps
                    activations, _ = activation_models[activation_model_idx](input_tensor=cam_input, target_category=int(pred_label))
                    
                    if not(stage_idx):
                        all_activations = activations
                    else:
                        all_activations = np.append(all_activations, activations, axis=0)
          
                ###################################################################
                ############# Combine Activaiton Maps and Groundtruth #############
                ###################################################################
                
                labels = np.reshape(gt_img,(gt_img.shape[0]*gt_img.shape[1]))
                
                test_data = np.zeros((all_activations.shape[0],all_activations.shape[1]*all_activations.shape[2]))
                
                for idx in range(all_activations.shape[0]):
                    test_data[idx,:] = np.reshape(all_activations[idx,:,:],
                        (1,all_activations[idx,:,:].shape[0]*all_activations[idx,:,:].shape[1]))
            
                test_data = test_data[feature_set,:]
                
                ## Test classifier (test_data, labels)
                pred_img = clf.predict(test_data.T)
                pred_img = pred_img.astype(int)
                
                ##################### Evaluate segmentation ###################
                intersection = np.logical_and(pred_img, labels)
                union = np.logical_or(pred_img, labels)
                
                ## Catch divide by zero(union of prediction and groundtruth is empty)
                try:
                    iou_score = np.sum(intersection) / np.sum(union)
                except:
                    iou_score = 0
                    
                metric_iou[sample_idx] = round(iou_score,5)
                      
            ############### Combine results from all samples ##################
            metric_iou_mean = round(np.mean(metric_iou),3)   
            metric_iou_std = round(np.std(metric_iou),3)   

            details = {'num_features':len(feature_set), 'subset': feature_set,'miou': metric_iou_mean, 'std': metric_iou_std}
            
            ## Save results to global dictionary of results
            model_name = str(len(feature_set))
            results_dict[model_name] = details
            
            ## Incrementally save results
            if (parameters.fitness_function == 'miou'):
                results_file = parameters.outf.replace('output','feature_ranking') + '/results_dict_miou_ranking.txt'
                results_savepath = parameters.outf.replace('output','feature_ranking') + '/results_dict_miou_ranking.npy'
            elif (parameters.fitness_function == 'importance'):
                results_file = parameters.outf.replace('output','feature_ranking') + '/results_dict_importance_ranking.txt'
                results_savepath = parameters.outf.replace('output','feature_ranking') + '/results_dict_importance_ranking.npy'
            
            ## Write results to text file
            with open(results_file, 'a+') as f:
                for key, value in details.items():
                    f.write('%s:%s\n' % (key, value))
                f.write('\n')
                f.close()
            
            ## Save results as numpy file
            np.save(results_savepath, results_dict, allow_pickle=True)
    
            del activations
            del clf
            del all_activations
            del details
    
    ###########################################################################
    ########################## Visualize Feature Ranking ######################
    ###########################################################################
    if parameters.feature_ranking_visualization:  
        
        ## Load ranked feature order
        if (parameters.fitness_function == 'miou'):
            idx_path = parameters.outf.replace('output','feature_ranking') + '/sorted_feature_idx_miou.npy'
            fitness_path = parameters.outf.replace('output','feature_ranking') + '/sorted_feature_fitness_miou.npy'
            results_file = parameters.outf.replace('output','feature_ranking') + '/results_dict_miou_ranking.txt'
            results_savepath = parameters.outf.replace('output','feature_ranking') + '/results_dict_miou_ranking.npy'
            method = 'mIoU'
            title = 'Pseudo-Label: mIoU vs Number of Features'
            figure_savepath = parameters.outf.replace('output','feature_ranking') + '/feature_ranking_miou.png'
            plt_color = 'blue'
            shade_color = 'lightskyblue'
        elif (parameters.fitness_function == 'importance'):
            idx_path = parameters.outf.replace('output','feature_ranking') + '/sorted_feature_idx_importance.npy'
            fitness_path = parameters.outf.replace('output','feature_ranking') + '/sorted_feature_fitness_miou_importance.npy'
            results_file = parameters.outf.replace('output','feature_ranking') + '/results_dict_importance_ranking.txt'
            results_savepath = parameters.outf.replace('output','feature_ranking') + '/results_dict_importance_ranking.npy'
            method = 'CAM Importance'
            title = 'CAM Importance: mIoU vs Number of Features'
            figure_savepath = parameters.outf.replace('output','feature_ranking') + '/feature_ranking_importance.png'
            plt_color = 'darkorange'
            shade_color = 'burlywood'
        
        sorted_idx = np.load(idx_path, allow_pickle=True)
        sorted_fitness_values = np.load(fitness_path, allow_pickle=True)
        results = np.load(results_savepath, allow_pickle=True).item()
        
        num_features = len(results)
        x_range = np.arange(1,num_features+1)
        y_range_mean = np.zeros(num_features)
        y_range_std = np.zeros(num_features)
        for idx, key in enumerate(results):
            y_range_mean[idx] = results[key]['miou']
            y_range_std[idx] = results[key]['std']
      
        ## Get legend entry
        max_val = np.max(y_range_mean)
        max_idx = np.where(y_range_mean==max_val)[0][0]
        max_val = round(max_val,4)
        
        legend_entry = 'Max mIoU: ' + str(max_val) + ', Num Features: ' + str(max_idx+1)
        
        
        plt.figure()
        
        plt.plot(x_range, y_range_mean, marker='^',color=plt_color)
        plt.fill_between(x_range, y_range_mean-y_range_std, y_range_mean+y_range_std, color=shade_color, alpha=0.5)
        plt.xlabel('Number of Features', fontsize = 12)
        plt.ylabel('mIoU', fontsize = 12)
        plt.title(title, fontsize = 14)
        plt.xlim((1,num_features))
        plt.ylim((0.15,0.5))
        
        
        plt.legend([legend_entry],loc='upper right')
        
        plt.savefig(figure_savepath)
        
        plt.close()
        
    ###########################################################################
    ########################## Visualize Feature Ranking ######################
    ###########################################################################
    if parameters.plot_top_activations:  
        
        ## Load ranked feature order
        if (parameters.fitness_function == 'miou'):
            idx_path = parameters.outf.replace('output','feature_ranking') + '/sorted_feature_idx_miou.npy'
            fitness_path = parameters.outf.replace('output','feature_ranking') + '/sorted_feature_fitness_miou.npy'
            results_file = parameters.outf.replace('output','feature_ranking') + '/results_dict_miou_ranking.txt'
            results_savepath = parameters.outf.replace('output','feature_ranking') + '/results_dict_miou_ranking.npy'
            method = 'mIoU'
            title = 'Pseudo-Label: mIoU vs Number of Features'
            figure_savepath = parameters.outf.replace('output','feature_ranking') + '/miou_activations'
            plt_color = 'blue'
            shade_color = 'lightskyblue'
        elif (parameters.fitness_function == 'importance'):
            idx_path = parameters.outf.replace('output','feature_ranking') + '/sorted_feature_idx_importance.npy'
            fitness_path = parameters.outf.replace('output','feature_ranking') + '/sorted_feature_fitness_miou_importance.npy'
            results_file = parameters.outf.replace('output','feature_ranking') + '/results_dict_importance_ranking.txt'
            results_savepath = parameters.outf.replace('output','feature_ranking') + '/results_dict_importance_ranking.npy'
            method = 'CAM Importance'
            title = 'CAM Importance: mIoU vs Number of Features'
            figure_savepath = parameters.outf.replace('output','feature_ranking') + '/importance_activations'
            plt_color = 'darkorange'
            shade_color = 'burlywood'
        
        sorted_idx = np.load(idx_path, allow_pickle=True)
        sorted_fitness_values = np.load(fitness_path, allow_pickle=True)
        results = np.load(results_savepath, allow_pickle=True).item()
        
        num_features = len(results)
        x_range = np.arange(1,num_features+1)
        y_range_mean = np.zeros(num_features)
        y_range_std = np.zeros(num_features)
        for idx, key in enumerate(results):
            y_range_mean[idx] = results[key]['miou']
            y_range_std[idx] = results[key]['std']
      
        ## Get legend entry
        max_val = np.max(y_range_mean)
        max_idx = np.where(y_range_mean==max_val)[0][0]
        max_val = round(max_val,4)
        
        
        ##########################################################
        ###################### Plot Activations ##################
        ##########################################################
        
        feature_set = sorted_idx[0:max_idx+1]
        
        
        for sample_idx, data in enumerate(tqdm(test_loader)):
      
            images, labels, gt_images = data[0].to(device), data[1].to(device), data[2]
            
            ## Normalize input to the model
            cam_input = norm_image(images)
            
            ## Get predicted class label
            output = model(cam_input)
            pred_label = np.argmax(output.detach().cpu().numpy()[0,:])
            
            
            ###############################################################
            ################# Extract activation maps #####################
            ###############################################################
            
            for stage_idx, activation_model_idx in enumerate(activation_models):
            
                ## Get activation maps
                activations, importance_weights = activation_models[activation_model_idx](input_tensor=cam_input, target_category=int(pred_label))
                
                if not(stage_idx):
                    all_activations = activations
                else:
                    all_activations = np.append(all_activations, activations, axis=0)
      
            ###############################################################
            ###################### Plot Activations #######################
            ###############################################################
            
             ## Visualize input and activations
            images = images.permute(2,3,1,0)
            images = images.detach().cpu().numpy()
            image = images[:,:,:,0]
            img = image
            
            plt.figure()
            plt.imshow(img)
            plt.axis('off')
            
            savepath = figure_savepath + '/img_' + str(sample_idx) + '/input.png'
            plt.savefig(savepath)
            
            plt.close()
            
            activation_set = all_activations[feature_set,:,:]
            
            for idk in range(activation_set.shape[0]):
                plt.figure()
                plt.imshow(activation_set[idk,:,:])
                plt.axis('off')
                
                savepath = figure_savepath + '/img_' + str(sample_idx) + '/activation_' + str(idk) + '.png'
                plt.savefig(savepath)
                plt.close()
                
    ###########################################################################
    ########################## Visualize Feature Ranking ######################
    ###########################################################################
    if parameters.plot_ranked_features:  
        
        ## Load ranked feature order
        if (parameters.fitness_function == 'miou'):
            idx_path = parameters.outf.replace('output','feature_ranking') + '/sorted_feature_idx_miou.npy'
            fitness_path = parameters.outf.replace('output','feature_ranking') + '/sorted_feature_fitness_miou.npy'
            miou_fitness_path = parameters.outf.replace('output','feature_ranking') + '/sorted_feature_fitness_miou.npy'
            method = 'mIoU'
            title = 'mIoU Feature Ranking Performance'
            figure_savepath = parameters.outf.replace('output','feature_ranking') + '/independent_feature_ranking_miou.png'
            plt_color = 'blue'
            shade_color = 'lightskyblue'
        elif (parameters.fitness_function == 'importance'):
            idx_path = parameters.outf.replace('output','feature_ranking') + '/sorted_feature_idx_importance.npy'
            fitness_path = parameters.outf.replace('output','feature_ranking') + '/sorted_feature_fitness_miou_importance.npy'
            miou_fitness_path = parameters.outf.replace('output','feature_ranking') + '/miou_importance.npy'
            method = 'CAM Importance'
            title = 'CAM Importance Feature Ranking Performance'
            figure_savepath = parameters.outf.replace('output','feature_ranking') + '/independent_feature_ranking_importance.png'
            figure_savepath_importance = parameters.outf.replace('output','feature_ranking') + '/independent_feature_ranking_importance_miou.png'
            plt_color = 'darkorange'
            shade_color = 'burlywood'
        
        sorted_idx = np.load(idx_path, allow_pickle=True)
        sorted_fitness_values = np.load(fitness_path, allow_pickle=True)
        miou_values = np.load(miou_fitness_path, allow_pickle=True)
        
        num_features = len(sorted_fitness_values)
        x_range = np.arange(1,num_features+1)
        y_range_mean = sorted_fitness_values
      
      
        ## Get legend entry
        max_val = np.max(y_range_mean)
        max_idx = np.where(y_range_mean==max_val)[0][0]
        max_val = round(max_val,4)

        plt.figure()
        
        plt.plot(x_range, y_range_mean, marker='^',color=plt_color)
        plt.xlabel('Index of Ranked Features', fontsize = 12)
       
        plt.title(title, fontsize = 14)
        plt.xlim((1,num_features))
        
        if (parameters.fitness_function == 'miou'):
            plt.ylabel('mIoU', fontsize = 12)
            plt.ylim((0.0,0.3))
            legend_entry = 'Max mIoU: ' + str(max_val)
            
        elif (parameters.fitness_function == 'importance'):
             plt.ylabel('Importance Weight', fontsize = 12)
             legend_entry = 'Max Importance: ' + str(max_val)
        
        plt.legend([legend_entry],loc='upper right')
        
        plt.savefig(figure_savepath)
        
        plt.close() 
        
        ##############################################################
        plt.figure()
        
        plt.plot(x_range, miou_values, marker='^',color=plt_color)
#        plt.fill_between(x_range, y_range_mean-y_range_std, y_range_mean+y_range_std, color=shade_color, alpha=0.5)
        plt.xlabel('Index of Ranked Features', fontsize = 12)
       
        plt.title(title, fontsize = 14)
        plt.xlim((1,num_features))
        
        if (parameters.fitness_function == 'miou'):
            plt.ylabel('mIoU', fontsize = 12)
            plt.ylim((0.0,0.3))
            
        elif (parameters.fitness_function == 'importance'):
            plt.ylabel('mIoU', fontsize = 12)
            plt.ylim((0.0,0.3))
             
            ## Get legend entry
            max_val = np.max(miou_values)
            max_idx = np.where(miou_values==max_val)[0][0]
            max_val = round(max_val,4)
            
            legend_entry = 'Max mIoU: ' + str(max_val)
        
        plt.legend([legend_entry],loc='upper right')
        
        plt.savefig(figure_savepath_importance)
        
        plt.close() 
        
        
    ###########################################################################
    ################## Get mIoU from Importance-ranked Features ###############
    ###########################################################################
    if parameters.miou_of_importance_ranks:
        
        idx_path = parameters.outf.replace('output','feature_ranking') + '/sorted_feature_idx_importance.npy'
        sorted_idx = np.load(idx_path, allow_pickle=True)
        
        num_pos_samples = 0
        
        for idx, data in enumerate(tqdm(train_loader)):
            
            ## Load sample and groundtruth
            images, labels, gt_images = data[0].to(device), data[1].to(device), data[2]
            
            ###############################################################
            ################### Get Pseudo-groundtruth ####################
            ###############################################################
            
            pred_label = int(labels.detach().cpu().numpy())
            
            if pred_label:
            
                gt_input = norm_image(images)
                
                gt_img = groundtruth_model(input_tensor=gt_input, target_category=int(pred_label))
                gt_img = gt_img[0, :]
                
                gt_thresh = threshold_otsu(gt_img)
                gt_img = gt_img > gt_thresh
                
                
                ###############################################################
                ################# Extract activation maps #####################
                ###############################################################
                
                ## Normalize input to the model
                cam_input = norm_image(images)
                
                for stage_idx, activation_model_idx in enumerate(activation_models):
                
                    ## Get activation maps
                    activations, _ = activation_models[activation_model_idx](input_tensor=cam_input, target_category=int(pred_label))
                    
                    if not(stage_idx):
                        all_activations = activations
                    else:
                        all_activations = np.append(all_activations, activations, axis=0)
                    
                ###################################################################
                ###################### Evaluate Fitness ###########################
                ###################################################################
                
                ## Initialize matrix [num_samples, num_features] to hold fitness values
                num_features = all_activations.shape[0]
        
          
        
                ## Evaluate fitness for each feature
                sample_fitness = np.zeros(num_features)
                for idk in range(num_features):
                    
                    ## Binarize feature
                    try:
                        img_thresh = threshold_otsu(all_activations[idk,:,:])
                        binary_feature_map = all_activations[idk,:,:] > img_thresh
                    except:
                        binary_feature_map = all_activations[idk,:,:] < 0.1
        
                    ## Compute fitness as IoU to pseudo-groundtruth
                    intersection = np.logical_and(binary_feature_map, gt_img)
                    union = np.logical_or(binary_feature_map, gt_img)
                    
                    ## Catch divide by zero(union of prediction and groundtruth is empty)
                    try:
                        iou_score = np.sum(intersection) / np.sum(union)
                    except:
                        iou_score = 0
                        
                    sample_fitness[idk] = iou_score
                        
               
                
                ## Add values to global matrix
                sample_fitness = sample_fitness[sorted_idx] ## Sort according to importance weights
                sample_fitness = np.expand_dims(sample_fitness,axis=0)
                if not(num_pos_samples):
                    fitness_values = sample_fitness
                else:
                    fitness_values = np.concatenate((fitness_values, sample_fitness), axis=0)
                
#                fitness_values[num_pos_samples,:] = sample_fitness
        
                num_pos_samples += 1
                
        
        ## Rank features
        avg_fitness_values = np.mean(fitness_values, axis = 0)
        fitness_path = parameters.outf.replace('output','feature_ranking') + '/miou_importance'
        
        np.save(fitness_path,avg_fitness_values)
        
            
  

    
    