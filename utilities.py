#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 13:07:24 2022

@author: cmccurley
"""

######################################################################
######################### Import Packages ############################
######################################################################
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets
from torch.utils.data import random_split
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from dataloaders import loaderPASCAL, loaderDSIAC

######################################################################
######################### Define Transforms ##########################
######################################################################
def define_transforms(parameters):
    
    if (parameters.run_mode == 'cam') or (parameters.run_mode == 'test-cams') or (parameters.run_mode == 'evaluate_cam_faithfulness'):
        
        if (parameters.DATASET == 'dsiac'):
            transform = transforms.Compose([transforms.ToTensor()])
            
        elif (parameters.DATASET == 'mnist'):
            transform = transforms.Compose([transforms.Resize(size=256),
                                            transforms.CenterCrop(224),
                                            transforms.Grayscale(3),
                                            transforms.ToTensor()])
        else:
            transform = transforms.Compose([transforms.ToTensor(), 
                                        transforms.Resize(256), 
                                        transforms.CenterCrop(224)])         
    else:
        if (parameters.DATASET == 'dsiac'):
            transform = transforms.Compose([transforms.ToTensor(), 
                                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        elif (parameters.DATASET == 'mnist'):
            parameters.MNIST_MEAN = (0.1307,)
            parameters.MNIST_STD = (0.3081,)
            transform = transforms.Compose([transforms.Resize(size=256),
                                            transforms.CenterCrop(224),
                                            transforms.Grayscale(3),
                                            transforms.ToTensor(), 
                                            transforms.Normalize(parameters.MNIST_MEAN, parameters.MNIST_STD)])
        else:
            transform = transforms.Compose([transforms.ToTensor(), 
                                            transforms.Resize(256), 
                                            transforms.CenterCrop(224), 
                                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    
    target_transform = transforms.Compose([transforms.ToTensor()])
    
    return transform, target_transform, parameters

######################################################################
############### Define Transform when Visualizing CAMs ###############
######################################################################
def cam_model_transforms(parameters):
    
    if (parameters.DATASET == 'mnist'):
        
        parameters.MNIST_MEAN = (0.1307,)
        parameters.MNIST_STD = (0.3081,)
        transform = transforms.Normalize(parameters.MNIST_MEAN, parameters.MNIST_STD)
        
    else:
        transform = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    
    return transform

######################################################################
######################### Define Dataloaders #########################
######################################################################
def define_dataloaders(transform, target_transform, parameters):
    
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
                                                  batch_size=parameters.TEST_BATCH_SIZE, 
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
        test_loader = torch.utils.data.DataLoader(testset, batch_size=parameters.TEST_BATCH_SIZE, shuffle=True, pin_memory=False)
        
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
        test_loader = torch.utils.data.DataLoader(testset, batch_size=parameters.TEST_BATCH_SIZE, shuffle=True, pin_memory=False)
        
    elif (parameters.DATASET == 'mnist'):
        
        parameters.mnist_data_path = '/home/UFAD/cmccurley/Public_Release/Army/data/mnist/'
        parameters.mnist_image_path = '/home/UFAD/cmccurley/Public_Release/Army/data/mnist/'
        parameters.mnist_gt_path = '/home/UFAD/cmccurley/Public_Release/Army/data/mnist/'
        
        classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
         
#        parameters.all_classes = {'aeroplane':1, 'bicycle':2, 'bird':3, 'boat':4,
#        'bottle':5, 'bus':6, 'car':7, 'cat':8, 'chair':9,
#        'cow':10, 'diningtable':11, 'dog':12, 'horse':13,
#        'motorbike':14, 'person':15, 'pottedplant':16,
#        'sheep':17, 'sofa':18, 'train':19, 'tvmonitor':20}
#        
#        classes = parameters.bg_classes + parameters.target_classes
#        
        parameters.NUM_CLASSES = len(classes)
#        
#        parameters.pascal_im_size = (224,224)
        
        trainset = datasets.MNIST(parameters.mnist_data_path, train=True,download=True,transform=transform)
        validset = datasets.MNIST(parameters.mnist_data_path, train=False,download=True,transform=transform)
        testset = datasets.MNIST(parameters.mnist_data_path, train=False,download=True,transform=transform)
        
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=parameters.BATCH_SIZE, shuffle=True, pin_memory=False)
        valid_loader = torch.utils.data.DataLoader(validset, batch_size=parameters.BATCH_SIZE, shuffle=True, pin_memory=False) 
        test_loader = torch.utils.data.DataLoader(testset, batch_size=parameters.TEST_BATCH_SIZE, shuffle=True, pin_memory=False)

    return train_loader, valid_loader, test_loader, classes, parameters

