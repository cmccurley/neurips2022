#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 13:47:01 2022

@author: cmccurley
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 13:21:04 2022

@author: cmccurley
"""

######################################################################
######################### Import Packages ############################
######################################################################
from tqdm import tqdm

# PyTorch packages
import torch
import torch.nn as nn
import torch.optim as optim

## Custom packages

######################################################################
#################### Define Training Script ##########################
######################################################################

def test_model(model, test_loader, classes, device, parameters):

    print('================== Testing ==================\n')
    
    model.eval()
    
    ############################# Total Accuracy #########################
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i,data in enumerate(tqdm(test_loader)):
            images, labels = data[0].to(device), data[1].to(device)
            
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    #                images = images.permute(2,3,1,0)
    #                images = images.detach().cpu().numpy()
    #                image = images[:,:,:,0]
    #                plt.imshow(image)
    #                plt.axis('off')
    #                
    #                if (labels.detach().cpu().numpy() == 1):
    #                    savename = parameters.outf + '/dataimages/aeroplane_' + str(i) 
    #                    
    #                elif (labels.detach().cpu().numpy() == 0):
    #                    savename = parameters.outf + '/dataimages/cat' + str(i)
    #                
    #                plt.savefig(savename)
    #                plt.close()
    
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    
    ######################### Per-class Performance ######################
    
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    
    with torch.no_grad():
        for data in (tqdm(test_loader)):
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
    
    ## Print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy))
            
    return