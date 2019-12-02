#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 16:37:46 2019

@author: anup
"""
# =============================================================================
# Importing Libraries
# =============================================================================
import os
import numpy as np
import datetime
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
from torchvision import transforms
import network_01 as nw
from utils_01 import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device :", device)
# =============================================================================
# Defining Parameters
# =============================================================================
VERSION = 7
EPOCHS = 100
BATCH_SIZE = 8
LEARNING_RATE = 0.001
TRAIN_DATA_PATH = "/home/anup/work/multilabel/label_tool/dataset_single/data/"
TEST_DATA_PATH = "/home/anup/work/multilabel/label_tool/dataset_single/data/"

transformed_dataset = MultiLabelDataset(img_file = '/home/anup/work/multilabel/label_tool/train_dataset.pkl',
                                        root_dir = TRAIN_DATA_PATH,
                                        transform = transforms.Compose([ReSize((256, 256)),
                                                                       ImCrop(224),
                                                                       TensorConv()]))

train_data_loader = data.DataLoader(transformed_dataset, 
                                    batch_size=BATCH_SIZE, 
                                    shuffle=True,  
                                    num_workers=1)

transformed_test_dataset = MultiLabelDataset(img_file = '/home/anup/work/multilabel/label_tool/test_dataset.pkl',
                                        root_dir = TEST_DATA_PATH,
                                        transform = transforms.Compose([ReSize((256, 256)),
                                                                       ImCrop(224),
                                                                       TensorConv()]))
test_data_loader = data.DataLoader(transformed_test_dataset,
                                    batch_size=BATCH_SIZE, 
                                    shuffle=True,  
                                    num_workers=1)


print("="*50)

if __name__ == '__main__':

    print("Number of train samples: ", len(transformed_dataset))
    print("Number of test samples: ", len(transformed_test_dataset))
    print("Detected Classes are: ", transformed_dataset.class_to_idx) # classes are detected by folder structure
    class_dict = transformed_dataset.class_to_idx
    class_dict_reverse = {v: k for k, v in class_dict.items()}
    class_dict.update(class_dict_reverse)
    save_dict(class_dict, 'class_label_dict')

    model = nw.multi_label_vgg_model(training=True).to(device)
    optimizer = torch.optim.Adam([{'params': model.features.parameters()}, {'params': model.classifier.parameters(), 'lr': 1e-6}], lr=1e-7)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=5, verbose=True)
    loss_func = nn.BCEWithLogitsLoss()

    # Training and Testing
    train_loss_list = []
    test_loss_list = []
    start_time = datetime.datetime.now()
    for epoch in range(EPOCHS):
        model = model.train()
        print("Epoch No: ", epoch + 1)
        running_loss = 0.0
        for step_train, train_sample_batched in enumerate(train_data_loader):
            images_batch, labels_batch = train_sample_batched['image'], train_sample_batched['comp_vector']
            images_batch = images_batch.to(device)
            labels_batch = labels_batch.to(device)
            optimizer.zero_grad()
            output = model(images_batch)
            loss = loss_func(output, labels_batch)   
            loss.backward() 
            optimizer.step()
            running_loss += loss.item()
        if step_train > 0:
            train_loss_list.append(running_loss / step_train)
            print('[%d] Epoch loss: %.6f' %
              (epoch + 1, running_loss / step_train))
              
            test_running_loss = 0.0
            accuracy_prd_list = []
            accuracy_inp_list = []
            for step_test, test_sample_batched in enumerate(test_data_loader):
                model=model.eval()
                test_inputs, test_labels = test_sample_batched['image'], test_sample_batched['comp_vector']
                test_inputs = test_inputs.cuda()
                test_labels = test_labels.cuda()
                test_output = model(test_inputs)
                test_loss = loss_func(test_output, test_labels)
                test_running_loss += test_loss.item()
                test_output[test_output > 0] = 1
                test_output[test_output < 0] = 0
                accuracy_prd_list.extend(test_output.cpu().tolist())
                accuracy_inp_list.extend(test_labels.cpu().tolist())
                test_loss = 0.0
            test_loss_list.append(test_running_loss/step_test)
            scheduler.step(test_running_loss)
            print('[%d] Valid loss: %.6f' %
              (epoch + 1, test_running_loss/step_test))
            accuracy_prd_list = [item for sublist in accuracy_prd_list for item in sublist]
            accuracy_inp_list = [item for sublist in accuracy_inp_list for item in sublist]
            test_accuracy = []
            for i in range(len(accuracy_prd_list)):
                if accuracy_prd_list[i] == accuracy_inp_list[i]:
                    test_accuracy.append(1)
                else:
                    test_accuracy.append(0)
            print('[%d] Valid accuracy: %.2f' %
              (epoch + 1, round(100 * sum(test_accuracy)/len(accuracy_inp_list), 2)))
            print("="*50)
            print(" "*50)
finish_time = datetime.datetime.now()
time_diff = finish_time - start_time
print ("Training time: ", divmod(time_diff.total_seconds(), 60))
torch.save(model.state_dict(), 'model/model_single_v' + str(VERSION) + '.pt')
# Create count of the number of epochs
epoch_count = range(1, len(train_loss_list) + 1)

# Visualize loss history
plt.plot(epoch_count, train_loss_list, 'r--')
plt.plot(epoch_count, test_loss_list, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('/home/anup/work/multilabel/losses/' + str(VERSION) + '.jpg')
plt.show()



    
