#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 23:24:50 2019

@author: anup
"""


import pickle
with open('train_dataset.pkl', 'rb') as f:
    c= pickle.load(f)
with open('test_dataset.pkl', 'rb') as f:
    d= pickle.load(f)

from shutil import copyfile
for i in d:
    copyfile('dataset_single/new_data/'+i['image_name'], 'dataset_single/test_data/'+i['image_name'])
    
    
    