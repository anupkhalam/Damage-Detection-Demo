#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 11:58:19 2019

@author: anup
"""

import os
import pandas as pd
import glob

labels = list(os.listdir('/home/anup/002_work/007_dam_det/04_project/008_multilabel/dataset_single/test/'))
labels.sort()
labels.insert(0, 'image_name')
#dataset = pd.DataFrame(columns=labels)
root = 'dataset_single/test'
map_dict = {i:[] for i in labels}
for cat_ in labels[1:]:
    files = glob.glob(root + '/' + cat_ + '/*.jpg')
    if len(files) != 10:
        raise ValueError
    files = [i.split('/')[-1] for i in files]
    files.sort()
#    dataset['image_name'] = files
    for lab_ in labels[1:]:
        if cat_ == lab_:
            map_dict['image_name'].extend(files)
            map_dict[lab_].extend([1 for i in range(len(files))])
        else:
            map_dict[lab_].extend([0 for i in range(len(files))])
            
dataset = pd.DataFrame(map_dict)
#dataset.sort_values(['image_name'], axis=0, ascending=True, inplace=True)




#dataset = dataset.reset_index(drop=True)
dataset.to_excel('map_dict_test.xlsx', index=None)




