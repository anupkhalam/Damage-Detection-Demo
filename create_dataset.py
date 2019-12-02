#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 15:21:08 2019

@author: anup
"""

import pandas as pd
import json
import pickle

mode = 'test'
dataset = pd.read_excel('map_dict_' + mode + '.xlsx')
dataset_list = []
dataset_json = {}
dataset['image_name'] = dataset.image_name.apply(lambda x : \
       "_".join(x.split('_')[2:4]) + '/' + str(x))

for index, rows in dataset.iterrows():
    dataset_dict = dict()
    dataset_dict['image_name']  = rows.image_name.split('/')[-1]
    dataset_dict['features'] = [rows.ceiling_nodam,
                                rows.ceiling_dam,
                                0,
                                rows.gutter_nodam,
                                rows.gutter_dam,
                                0,
                                rows.roof_nodam,
                                rows.roof_dam,
                                0,
                                rows.stucco_nodam,
                                rows.stucco_dam,
                                0,
                                rows.vent_nodam,
                                rows.vent_dam,
                                0,0,0,0,0,0,0,0,0,0]
    dataset_list.append(dataset_dict)
    dataset_json[rows.image_name.split('/')[-1]] = dataset_dict

with open('dataset_list_'+ mode +'.pkl', 'wb') as handle:
    pickle.dump(dataset_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(mode + '_dataset.json', 'w') as json_file:
    json.dump(dataset_json, json_file)
    