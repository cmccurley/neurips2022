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
import util

######################################################################
#################### Define Training Script ##########################
######################################################################

def train_model(model, train_loader, valid_loader, logfilename, device, parameters):

    print('================= Training =================\n')
                
    ################ Define Model and Training Criterion #############
     
    ## Define loss function (BCE for binary classification)
    criterion = nn.CrossEntropyLoss()
    
    ## Define optimizer for weight updates
    optimizer = optim.Adamax(model.parameters(), lr=parameters.LR)
    
    ## Set to GPU
    if parameters.cuda:
        criterion.cuda()     
        model.cuda()
    
    ###################### Train the Network ########################
    
    ## Create file to display epoch loss
    f = open(logfilename, 'w')
    f.close()
    
    model.train()
    
    ## Train for the desired number of epochs
    print('Fine tuning model...')
    for epoch in range(0,parameters.EPOCHS+1):
    
        running_loss = 0.0
        for i, data in enumerate(tqdm(train_loader)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            # print statistics
            running_loss += loss.item()
            
        epoch_loss = running_loss/len(train_loader)       
        
        print(f'=========== Epoch: {epoch} ============')
        
        # Display training and validation loss to terminal and update loss file
        if not(epoch % parameters.update_on_epoch):
            model.eval()
            temptrain = parameters.outf + '/model_eoe_' + str(epoch) + '.pth'
            torch.save(model.state_dict(), temptrain)
            util.print_status(epoch, epoch_loss, train_loader, valid_loader, model, device, logfilename)
            model.train()
            
    return