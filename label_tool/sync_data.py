#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 16:06:24 2019

@author: anup
"""

import json
with open('dataset.json', 'rb') as f:
    data = json.load(f)

c= list(data.keys())

import glob
import os
image_list = glob.glob('/home/anup/work/multilabel/label_tool/dataset_single/new_data/*.jpg')
image_1_list = [k.split('/')[-1] for k in image_list]
e_list = [i for i in image_1_list if i not in c]
for i in e_list:
    os.remove('/home/anup/work/multilabel/label_tool/dataset_single/new_data/' + i)
    

d = [k for k in image_list if k.split('/')[-1] in e_list]
from shutil import copyfile
for i in d:
    copyfile(i, '/home/anup/work/multilabel/label_tool/dataset_single/fill_data/' + i.split('/')[-1])



    