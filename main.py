# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 13:46:23 2020

@author: cmccurley
"""

"""
***********************************************************************
    *  File:  main.py
    *  Name:  Connor H. McCurley
    *  Date:  2022-04
    *  Desc:  
    *
    *
    *  Author: Connor H. McCurley
    *  University of Florida, Electrical and Computer Engineering
    *  Email Address: cmccurley@ufl.edu
    *  Latest Revision: April 2022
    *  This product is Copyright (c) 2022 University of Florida
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
import json
import argparse
from tqdm import tqdm
import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn.metrics import precision_recall_fscore_support as prfs

# PyTorch packages
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

## Custom packages
import main_parameters
import initialize_network

from utilities import define_transforms, define_dataloaders, cam_model_transforms
from train_network import train_model
from test_network import test_model

from util import convert_gt_img_to_mask, cam_img_to_seg_img


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
    parameters = main_parameters.set_parameters(args)
    
    parameter_file = parameters.outf.replace('output','baseline_experiments') + '/parameters_null.txt'
    results_file = parameters.outf.replace('output','baseline_experiments') + '/results_eigencam.txt'
    
    ## Define files to save epoch training/validation
    logfilename = parameters.outf + parameters.loss_file
    
#    ## Read parameters
#    parser = argparse.ArgumentParser()
#    parameters = parser.parse_args()
#    with open(parameter_file, 'r') as f:
#        parameters.__dict__ = json.load(f)
    
    
    ######################################################################
    ################## Define data loaders and transforms ################
    ######################################################################

    ## Define data transforms
    transform, target_transform, parameters = define_transforms(parameters)
    
    ## Define dataloaders
    train_loader, valid_loader, test_loader, classes, parameters = define_dataloaders(transform, target_transform, parameters)
    
    ######################################################################
    ####################### Initialize Network ###########################
    ######################################################################
    ## Define model
    model = initialize_network.init(parameters)
    
    ## Save initial weights for further training
    temptrain = parameters.outf + parameters.parameter_path
    torch.save(model.state_dict(), temptrain)
    
    ## Detect if GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    if (parameters.run_mode == 'train'):
        ######################################################################
        ######################## Fine Tune Model #############################
        ######################################################################
        ## Train image-level classifier for desired number of epochs
        train_model(model, train_loader, valid_loader, logfilename, device, parameters)
       
    elif (parameters.run_mode == 'test'):
        ######################################################################
        ########################### Test Model ###############################
        ######################################################################
        ## Test image-level classifier
        test_model(model, test_loader, classes, device, parameters)
        

    elif (parameters.run_mode == 'cam'):
        ######################################################################
        ########################### Compute CAMs #############################
        ######################################################################
        
        import numpy as np
        from cam_functions import GradCAM, LayerCAM
        from cam_functions.utils.image import show_cam_on_image, preprocess_image

        ## Turn on gradients for CAM computation 
        for param in model.parameters():
            param.requires_grad = True
    
        model.eval()
        
        ## Define CAMs for ResNet18
        if (parameters.model == 'resnet18'):
#            cam0 = GradCAM(model=model, target_layers=[model.layer1[-1]], use_cuda=parameters.cuda)
#            cam1 = GradCAM(model=model, target_layers=[model.layer1[-1]], use_cuda=parameters.cuda)
#            cam2 = GradCAM(model=model, target_layers=[model.layer2[-1]], use_cuda=parameters.cuda)
#            cam3 = GradCAM(model=model, target_layers=[model.layer3[-1]], use_cuda=parameters.cuda)
#            cam4 = GradCAM(model=model, target_layers=[model.layer4[-1]], use_cuda=parameters.cuda)
            
            cam0 = LayerCAM(model=model, target_layers=[model.layer1[-1]], use_cuda=parameters.cuda)
            cam1 = LayerCAM(model=model, target_layers=[model.layer1[-1]], use_cuda=parameters.cuda)
            cam2 = LayerCAM(model=model, target_layers=[model.layer2[-1]], use_cuda=parameters.cuda)
            cam3 = LayerCAM(model=model, target_layers=[model.layer3[-1]], use_cuda=parameters.cuda)
            cam4 = LayerCAM(model=model, target_layers=[model.layer4[-1]], use_cuda=parameters.cuda)
    
        elif (parameters.model == 'vgg16'):
#            cam0 = GradCAM(model=model, target_layers=[model.features[4]], use_cuda=parameters.cuda)
#            cam1 = GradCAM(model=model, target_layers=[model.features[9]], use_cuda=parameters.cuda)
#            cam2 = GradCAM(model=model, target_layers=[model.features[16]], use_cuda=parameters.cuda)
#            cam3 = GradCAM(model=model, target_layers=[model.features[23]], use_cuda=parameters.cuda)
#            cam4 = GradCAM(model=model, target_layers=[model.features[30]], use_cuda=parameters.cuda)
            
            cam0 = LayerCAM(model=model, target_layers=[model.features[4]], use_cuda=parameters.cuda)
            cam1 = LayerCAM(model=model, target_layers=[model.features[9]], use_cuda=parameters.cuda)
            cam2 = LayerCAM(model=model, target_layers=[model.features[16]], use_cuda=parameters.cuda)
            cam3 = LayerCAM(model=model, target_layers=[model.features[23]], use_cuda=parameters.cuda)
            cam4 = LayerCAM(model=model, target_layers=[model.features[30]], use_cuda=parameters.cuda)
        
        
        if (parameters.DATASET == 'mnist'):
            parameters.MNIST_MEAN = (0.1307,)
            parameters.MNIST_STD = (0.3081,)
            norm_image = transforms.Normalize(parameters.MNIST_MEAN, parameters.MNIST_STD)
            gt_transform = transforms.Grayscale(1)
            
        else:
            norm_image = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    
        target_category = None
        img_idx = 0
        for data in tqdm(test_loader):
            
                if (parameters.DATASET == 'mnist'):
                    images, labels = data[0].to(device), data[1].to(device)
                else:
                    images, labels, gt_images = data[0].to(device), data[1].to(device), data[2]
                
                ## Normalize input to the model
                cam_input = norm_image(images)
                
                ## Get predicted class label
                output = model(cam_input)
                pred_label = np.argmax(output.detach().cpu().numpy()[0,:])
                
#                pred_label = int(labels.detach().cpu().numpy())
#                if pred_label:
#                    pred_label = 0
#                elif not(pred_label):
#                    pred_label = 1
           
                ## Get CAMs
                grayscale_cam0 = cam0(input_tensor=cam_input,
                                target_category=int(pred_label))
                grayscale_cam1 = cam1(input_tensor=cam_input,
                                target_category=int(pred_label))
                grayscale_cam2 = cam2(input_tensor=cam_input,
                                target_category=int(pred_label))
                grayscale_cam3 = cam3(input_tensor=cam_input,
                                target_category=int(pred_label))
                grayscale_cam4 = cam4(input_tensor=cam_input,
                                target_category=int(pred_label))

                # Here grayscale_cam has only one image in the batch
                grayscale_cam0 = grayscale_cam0[0, :]
                grayscale_cam1 = grayscale_cam1[0, :]
                grayscale_cam2 = grayscale_cam2[0, :]
                grayscale_cam3 = grayscale_cam3[0, :]
                grayscale_cam4 = grayscale_cam4[0, :]
                
                ## Visualize input and CAMs
                images = images.permute(2,3,1,0)
                images = images.detach().cpu().numpy()
                image = images[:,:,:,0]
                img = image
                
                if (parameters.DATASET == 'mnist'):
                    gt_img = Image.fromarray(np.uint8(img*255))
                    gt_images = gt_transform(gt_img)
                    gt_image = np.asarray(gt_images)/255
                    
                    gt_image[np.where(gt_image>0.2)] = 1
                    gt_image[np.where(gt_image<=0.1)] = 0
                    
                else:
                    gt_images = gt_images.detach().cpu().numpy()
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
                
#                ## Show image and groundtruth
#                fig, axis = plt.subplots(nrows=1, ncols=2)
#                axis[0].imshow(img)
#                axis[1].imshow(gt_image, cmap='gray')
#                axis[0].axis('off')
#                axis[1].axis('off')

                ## Plot CAMs (VGG16)
                fig, axis = plt.subplots(nrows=1, ncols=6)
                
                axis[0].imshow(img) 
    
                cam_image0 = show_cam_on_image(img, grayscale_cam0, True)
                cam_image1 = show_cam_on_image(img, grayscale_cam1, True)
                cam_image2 = show_cam_on_image(img, grayscale_cam2, True)
                cam_image3 = show_cam_on_image(img, grayscale_cam3, True)
                cam_image4 = show_cam_on_image(img, grayscale_cam4, True)
                    
                axis[1].imshow(cam_image0, cmap='jet')
                axis[2].imshow(cam_image1, cmap='jet')
                axis[3].imshow(cam_image2, cmap='jet')
                axis[4].imshow(cam_image3, cmap='jet')
                axis[5].imshow(cam_image4, cmap='jet')
                
                axis[0].axis('off')
                axis[1].axis('off')
                axis[2].axis('off')
                axis[3].axis('off')
                axis[4].axis('off')
                axis[5].axis('off')
                
#                title = 'Pred: ' + classes[pred_label]
                title = 'Input'
                axis[0].set_title(title)
                axis[1].set_title('Layer 1')
                axis[2].set_title('Layer 2')
                axis[3].set_title('Layer 3')
                axis[4].set_title('Layer 4')
                axis[5].set_title('Layer 5')
                title2 = 'Pred: ' + classes[pred_label] + ' / Actual: ' + classes[int(labels[0].detach().cpu())]
                fig.suptitle(title2)
                
#                ## Plot CAMs (ResNet18)
#                fig, axis = plt.subplots(nrows=1, ncols=5)
#                
#                axis[0].imshow(img) 
#    
#                cam_image1 = show_cam_on_image(img, grayscale_cam1, True)
#                cam_image2 = show_cam_on_image(img, grayscale_cam2, True)
#                cam_image3 = show_cam_on_image(img, grayscale_cam3, True)
#                cam_image4 = show_cam_on_image(img, grayscale_cam4, True)
#                    
#                axis[1].imshow(cam_image1, cmap='jet')
#                axis[2].imshow(cam_image2, cmap='jet')
#                axis[3].imshow(cam_image3, cmap='jet')
#                axis[4].imshow(cam_image4, cmap='jet')
#                
#                axis[0].axis('off')
#                axis[1].axis('off')
#                axis[2].axis('off')
#                axis[3].axis('off')
#                axis[4].axis('off')
#                
##                title = 'Pred: ' + classes[pred_label]
#                title = 'Input'
#                axis[0].set_title(title)
#                axis[1].set_title('Layer 1')
#                axis[2].set_title('Layer 2')
#                axis[3].set_title('Layer 3')
#                axis[4].set_title('Layer 4')
#               
       
                savename = parameters.outf
                savename = savename.replace('output','cams/layercam/cams/')
                savename = savename + 'img_' + str(img_idx)
                fig.savefig(savename)
                
                plt.close()
                
                img_idx += 1
                
                if not(img_idx%25):
                    print(str(img_idx))
                    
                
    elif (parameters.run_mode == 'test-cams'):
        ######################################################################
        ########################### Compute CAMs #############################
        ######################################################################
        import numpy as np
        import matplotlib.pyplot as plt
        from cam_functions import GradCAM, LayerCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, EigenCAM
        from cam_functions.utils.image import show_cam_on_image, preprocess_image
        
        results_dict = dict()
        visualization_results = dict()
        visualization_results['thresholds'] = parameters.CAM_SEG_THRESH
        
        norm_image = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        
        methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "eigencam": EigenCAM,
         "layercam": LayerCAM}
        
        if (parameters.model == 'vgg16'):
            cam_layers = \
            {4:"layer1",
             9:"layer2",
             16:"layer3",
             23:"layer4",
             30:"layer5"}

        ## Turn on gradients for CAM computation 
        for param in model.parameters():
            param.requires_grad = True
    
        model.eval()
        
        for current_cam in parameters.cams:
            for layer in parameters.layers:      
                for thresh in parameters.CAM_SEG_THRESH:
                
                    n_samples = len(test_loader)
                    metric_iou = np.zeros(n_samples, dtype="float32")
                    metric_precision = np.zeros(n_samples, dtype="float32")
                    metric_recall = np.zeros(n_samples, dtype="float32")
                    metric_f1_score = np.zeros(n_samples, dtype="float32")
                    
                    current_model = current_cam + '_' + cam_layers[layer]
                    print(current_model+'_thresh_'+str(thresh))
                
                    cam_algorithm = methods[current_cam]
                
                    if (parameters.model == 'vgg16'):
                        cam = cam_algorithm(model=model, target_layers=[model.features[layer]], use_cuda=parameters.cuda)
        
                    target_category = None
                    
                    idx = 0
                    for data in tqdm(test_loader):
                  
                            images, labels, gt_images = data[0].to(device), data[1].to(device), data[2]
                            
                            ############# Convert groundtruth image into mask #############
                            gt_img = convert_gt_img_to_mask(gt_images, labels, parameters)
                            
                            ## Normalize input to the model
                            cam_input = norm_image(images)
                            
                            ##################### Get CAM Segmentaiton ####################
                            grayscale_cam = cam(input_tensor=cam_input, target_category=int(labels))
                            grayscale_cam = grayscale_cam[0, :]
                            
                            ## Binarize CAM for segmentation
                            pred_img = cam_img_to_seg_img(grayscale_cam, thresh)
                            
                            ##################### Evaluate segmentation ###################
                            intersection = np.logical_and(pred_img, gt_img)
                            union = np.logical_or(pred_img, gt_img)
                            
                            ## Catch divide by zero(union of prediction and groundtruth is empty)
                            try:
                                iou_score = np.sum(intersection) / np.sum(union)
                            except:
                                iou_score = 0
                                
                            metric_iou[idx] = round(iou_score,5)
                            
                            
#                            prec, rec, f1, _ = prfs(gt_img.reshape(gt_img.shape[0]*gt_img.shape[1]), 
#                                                    pred_img.reshape(pred_img.shape[0]*pred_img.shape[1]),
#                                                    pos_label=1,
#                                                    average='binary') 
                            prec = 0
                            rec = 0
                            f1 = 0
                            
                            metric_precision[idx], metric_recall[idx], metric_f1_score[idx] = round(prec,5), round(rec,5), round(f1,5)
                            
                            idx +=1
                            
                            images.detach()
                            labels.detach()
                            gt_images.detach()
                            cam_input.detach()
                            
                            
                    ## Compute statistics over the entire test set
                    metric_iou_mean = round(np.mean(metric_iou),3)
                    metric_iou_std = round(np.std(metric_iou),3)
                    metric_precision_mean = round(np.mean(metric_precision),3)
                    metric_precision_std = round(np.std(metric_precision),3)
                    metric_recall_mean = round(np.mean(metric_recall),3)
                    metric_recall_std = round(np.std(metric_recall),3)
                    metric_f1_score_mean = round(np.mean(metric_f1_score),3)
                    metric_f1_score_std = round(np.std(metric_f1_score),3)
                    
                    
#                    ## Amalgamate results into one dictionary
#                    details = {'method': current_model,
#                               'threshold': thresh,
#                               'iou mean':metric_iou_mean,
#                               'iou std':metric_iou_std,
#                               'precision mean':metric_precision_mean,
#                               'precision std':metric_precision_std,
#                               'recall mean':metric_recall_mean,
#                               'recall std':metric_recall_std,
#                               'f1 score mean':metric_f1_score_mean,
#                               'f1 score std':metric_f1_score_std}
                    
                    ## Amalgamate results into one dictionary
                    details = {'method': current_model,
                               'threshold': thresh,
                               'iou mean':metric_iou_mean,
                               'iou std':metric_iou_std}
                    
                    ## Save results to global dictionary of results
                    model_name = current_model + '_thresh_' + str(thresh)
                    results_dict[model_name] = details
                    
                    ## Write results to text file
                    with open(results_file, 'a+') as f:
                        for key, value in details.items():
                            f.write('%s:%s\n' % (key, value))
                        f.write('\n')
                        f.close()
                    
                    ## Clean up memory
                    del details
                    del cam
                    del grayscale_cam
                    del metric_iou_mean
                    del metric_iou_std
                    del metric_precision_mean
                    del metric_precision_std
                    del metric_recall_mean
                    del metric_recall_std
                    del metric_f1_score_mean
                    del metric_f1_score_std
                    del metric_iou 
                    del metric_precision 
                    del metric_recall 
                    del metric_f1_score 
                    del cam_algorithm
          
                    torch.cuda.empty_cache()
        
                ## Save parameters         
                with open(parameter_file, 'w') as f:
                    json.dump(parameters.__dict__, f, indent=2)
                f.close 
                    
                ## Save results
                results_savepath = parameters.outf.replace('output','baseline_experiments') + '/results_eigencam.npy'
                np.save(results_savepath, results_dict, allow_pickle=True)
    
#    results_savepath = parameters.outf.replace('output','baseline_experiments_t_aeroplane_b_cat') + '/visualization_results.npy'
#    np.save(results_savepath, visualization_results, allow_pickle=True)
    
    
        
#                ## Visualize input and CAMs
#                images = images.permute(2,3,1,0)
#                images = images.detach().cpu().numpy()
#                image = images[:,:,:,0]
#                img = image
                
#                #######################################
#                ## Show image and groundtruth
#                fig, axis = plt.subplots(nrows=1, ncols=2)
#                axis[0].imshow(img)
#                axis[1].imshow(gt_image, cmap='gray')
#                axis[0].axis('off')
#                axis[1].axis('off')
#
#                ## Plot CAMs
#                fig, axis = plt.subplots(nrows=1, ncols=2)
#                axis[0].imshow(img) 
#                cam_image = show_cam_on_image(img, grayscale_cam, True)
#                axis[1].imshow(cam_image, cmap='jet')
#                ########################################
                
#                plt.figure()
#                plt.imshow(gt_image,cmap='gray')
#                plt.title('GT Mask')
#                
#                plt.figure()
#                plt.imshow(norm_cam,cmap='gray')
#                plt.title('CAM Segmentation')
    
    elif ( parameters.run_mode == 'evaluate_cam_faithfulness'):
        
        ######################################################################
        ##################### Evaluate CAM Faithfulness ######################
        ######################################################################
        import numpy as np
        import matplotlib.pyplot as plt
        from cam_functions import GradCAM, LayerCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, EigenCAM
        from cam_functions.utils.image import show_cam_on_image, preprocess_image
        
        results_dict = dict()
        visualization_results = dict()
              
        norm_image = cam_model_transforms(parameters)
        
        methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "eigencam": EigenCAM,
         "layercam": LayerCAM}
        
        if (parameters.model == 'vgg16'):
            cam_layers = \
            {4:"layer1",
             9:"layer2",
             16:"layer3",
             23:"layer4",
             30:"layer5"}

        ## Turn on gradients for CAM computation 
        for param in model.parameters():
            param.requires_grad = True
    
        model.eval()
        
        for current_cam in parameters.cams:
            for layer in parameters.layers:      
                
                n_samples = len(test_loader)
#                metric_iou = np.zeros(n_samples, dtype="float32")
#                metric_precision = np.zeros(n_samples, dtype="float32")
#                metric_recall = np.zeros(n_samples, dtype="float32")
#                metric_f1_score = np.zeros(n_samples, dtype="float32")
                
                current_model = current_cam + '_' + cam_layers[layer]
#                print(current_model+'_thresh_'+str(thresh))
            
                cam_algorithm = methods[current_cam]
            
                if (parameters.model == 'vgg16'):
                    cam = cam_algorithm(model=model, target_layers=[model.features[layer]], use_cuda=parameters.cuda)
    
                target_category = None
                
                idx = 0
                for data in tqdm(test_loader):
              
#                    images, labels, gt_images = data[0].to(device), data[1].to(device), data[2]
                    
                    images, labels = data[0].to(device), data[1].to(device)
                    
                    ## Normalize input to the model
                    model_input = norm_image(images)
                    
                    ## Get estimated image-level label
                    
#                    criterion = nn.CrossEntropyLoss()
#                    loss = criterion(outputs, labels)
                    
                    pred_label = model(model_input).argmax()
                    
                    ####################### Get CAM Heatmap #######################
                    grayscale_cam = cam(input_tensor=model_input, target_category=int(pred_label))
                    grayscale_cam = grayscale_cam[0, :]
                    
                    ## Visualize input and CAMs
                    images = images.permute(2,3,1,0)
                    images = images.detach().cpu().numpy()
                    image = images[:,:,:,0]
                    img = image
                    
                    ## Plot CAMs
                    fig, axis = plt.subplots(nrows=1, ncols=2)
                    axis[0].imshow(img) 
                    cam_image = show_cam_on_image(img, grayscale_cam, True)
                    axis[1].imshow(cam_image, cmap='jet')
            
                    
        
                        
#                    metric_iou[idx] = round(iou_score,5)
#                    
#                    
#    #                            prec, rec, f1, _ = prfs(gt_img.reshape(gt_img.shape[0]*gt_img.shape[1]), 
#    #                                                    pred_img.reshape(pred_img.shape[0]*pred_img.shape[1]),
#    #                                                    pos_label=1,
#    #                                                    average='binary') 
#                    prec = 0
#                    rec = 0
#                    f1 = 0
#                    
#                    metric_precision[idx], metric_recall[idx], metric_f1_score[idx] = round(prec,5), round(rec,5), round(f1,5)
#                    
#                    idx +=1
#                    
#                    images.detach()
#                    labels.detach()
#                    gt_images.detach()
#                    model_input.detach()
#                            
#                            
#                    ## Compute statistics over the entire test set
#                    metric_iou_mean = round(np.mean(metric_iou),3)
#                    metric_iou_std = round(np.std(metric_iou),3)
#                    metric_precision_mean = round(np.mean(metric_precision),3)
#                    metric_precision_std = round(np.std(metric_precision),3)
#                    metric_recall_mean = round(np.mean(metric_recall),3)
#                    metric_recall_std = round(np.std(metric_recall),3)
#                    metric_f1_score_mean = round(np.mean(metric_f1_score),3)
#                    metric_f1_score_std = round(np.std(metric_f1_score),3)
#                    
                    
#                    ## Amalgamate results into one dictionary
#                    details = {'method': current_model,
#                               'threshold': thresh,
#                               'iou mean':metric_iou_mean,
#                               'iou std':metric_iou_std,
#                               'precision mean':metric_precision_mean,
#                               'precision std':metric_precision_std,
#                               'recall mean':metric_recall_mean,
#                               'recall std':metric_recall_std,
#                               'f1 score mean':metric_f1_score_mean,
#                               'f1 score std':metric_f1_score_std}
                    
#                    ## Amalgamate results into one dictionary
#                    details = {'method': current_model,
#                               'threshold': thresh,
#                               'iou mean':metric_iou_mean,
#                               'iou std':metric_iou_std}
#                    
#                    ## Save results to global dictionary of results
#                    model_name = current_model + '_thresh_' + str(thresh)
#                    results_dict[model_name] = details
#                    
#                    ## Write results to text file
#                    with open(results_file, 'a+') as f:
#                        for key, value in details.items():
#                            f.write('%s:%s\n' % (key, value))
#                        f.write('\n')
#                        f.close()
#                    
#                    ## Clean up memory
#                    del details
#                    del cam
#                    del grayscale_cam
#                    del metric_iou_mean
#                    del metric_iou_std
#                    del metric_precision_mean
#                    del metric_precision_std
#                    del metric_recall_mean
#                    del metric_recall_std
#                    del metric_f1_score_mean
#                    del metric_f1_score_std
#                    del metric_iou 
#                    del metric_precision 
#                    del metric_recall 
#                    del metric_f1_score 
#                    del cam_algorithm
#          
#                    torch.cuda.empty_cache()
#        
#                ## Save parameters         
#                with open(parameter_file, 'w') as f:
#                    json.dump(parameters.__dict__, f, indent=2)
#                f.close 
#                    
#                ## Save results
#                results_savepath = parameters.outf.replace('output','baseline_experiments') + '/results_eigencam.npy'
#                np.save(results_savepath, results_dict, allow_pickle=True)
        
        
        
        