#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 10:38:14 2019

@author: anup
"""
import os
import cv2
import numpy as np
	
# Get the list of all files in directory tree at given path
train_list = list()
train_folder = '/home/anup/work/multilabel/dataset_single/train'
for (dirpath, dirnames, filenames) in os.walk(train_folder):
    train_list += [os.path.join(dirpath, file) for file in filenames if file.endswith('.jpg')]

test_list=[]
test_folder = '/home/anup/work/multilabel/dataset_single/test'
for (dirpath, dirnames, filenames) in os.walk(test_folder):
    test_list += [os.path.join(dirpath, file) for file in filenames if file.endswith('.jpg')]

duplicate_counter=0
duplicates=[]
for index, orginal in enumerate(train_list):
    print("{}/{}".format(index, len(train_list)))
    for duplicate in test_list:
        original_ = cv2.imread(orginal)
        duplicate_ = cv2.imread(duplicate)

        # 1) Check if 2 images are equals
        if original_.shape == duplicate_.shape:
            difference = cv2.subtract(original_, duplicate_)
            b, g, r = cv2.split(difference)
            if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
                duplicate_counter += 1
                duplicates.append((orginal, duplicate))
                print("The images are completely Equal")
        else:
            continue

import pickle
with open('duplicate_file.pkl', 'wb') as f:
    pickle.dump(c, f, pickle.HIGHEST_PROTOCOL)

import os

for i in c:
    try:
        os.remove(i[1])
    except:
        pass



