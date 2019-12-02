#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 09:50:12 2019

@author: anup
"""

#import pandas as pd
#import numpy as np

#dataset = pd.read_excel('/home/anup/002_work/007_dam_det/05_label_tool/Scoping App - Combinations.xlsx', 'Elevation')
#dataset = dataset.replace(np.nan, 'NA', regex=True)
#dataset = dataset.drop(['GROUP', 'SUB GROUP', 'LEVEL 1'], axis=1)
#cols = sorted(list(dataset))
#
#def build_profile(data, cols, item_profile):
#    for index, item in enumerate(cols):
#        print("####", item)
#        print("%%%%%", cols)
#        print(list(set(data[item].tolist())))
#        data_cols = list(set(data[item].tolist()))
#        if 'NA' in data_cols:
#            data_cols.remove('NA')
#        if data_cols != ['NA']:
#            for index_, level_ in enumerate(data_cols):
#                item_profile[level_] = []
#                sub_data = data.loc[data[item] == level_]
#                sub_data = sub_data.reset_index(drop=True)
#                if len(sub_data) == 1:
#                    if index + 1 < len(cols):
#                        if list(set(data[cols[index + 1]].tolist()))[0] == 'NA':
#                            item_profile[level_].append(level_)
#                        else:
#                            new_cols = cols[1:]
#                            new_data = sub_data[new_cols].copy()
#                            new_data = new_data.reset_index(drop=True)
#                            item_profile[level_].append(build_profile(new_data, new_cols, {}))
#                    elif index + 1 == len(cols):
#                        item_profile[level_].append(level_)
#                else:
#                    new_cols = cols[1:]
#                    new_data = sub_data[new_cols].copy()
#                    new_data = new_data.reset_index(drop=True)
#                    item_profile[level_].append(build_profile(new_data, new_cols, {}))
#                    print("^^^^^^^", item_profile)
#    return item_profile

#c = build_profile(dataset, cols, {})             
#            
            
        


import pymongo
        
    


item_profile = {}
for index, item in enumerate(sorted(list(set(dataset.Item.tolist())))):
    item_profile[item] = []
    for index_2, level in enumerate(cols):
        dataset.drop
        print(index, ' : ', item)
        sub_dataset = dataset.loc[dataset['Item'] == item]
        sub_dataset = sub_dataset.reset_index(drop=True)
        level_items = list(set(sub_dataset[level].tolist()))
        level_items.remove('NA')




