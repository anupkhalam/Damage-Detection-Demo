#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 15:29:22 2019

@author: anup
"""

"""
This routine will convert back the json created with peril non peril and additional cl
classes such as doors window and downsput to previous 5 classes with damaged / non-damaged status
"""
import pickle
import json
mode= 'test'

void = ['ad_20190716_gutter_dam_13.jpg',
        'an_20190724_gutter_dam_13.jpg',
        'an_20190724_gutter_dam_28.jpg',
        'ja_20190723_gutter_dam_22.jpg']

elevation_items = ['Ceiling',
                   'Gutter',
                   'Roof',
                   'Stucco',
                   'Vent']

elevation_rows = ['_'.join(i.lower().split()) for i in elevation_items]

damage_status_pre = ['No Damage','Peril Damage','Non Peril Damage']
damage_status = ['_'.join(i.lower().split()) for i in damage_status_pre]

dataset_rows = []
dataset_rows_dict = {}
for h, i in enumerate(elevation_rows):
    for k, j in enumerate(damage_status):
        dataset_rows.append(i + '_'+ j)
        dataset_rows_dict[i + '_'+ j] = [elevation_items[h], damage_status_pre[k]]


with open('labeller/'+ mode + '_dataset.json', 'r') as json_file:
    image_profile = json.load(json_file)
    for i in void:
        image_profile.pop(i,None)

k=[]

def new_vector(c):
    m = {}
    for i_1, i_2 in enumerate(dataset_rows):
        m[i_2] = c[i_1]
    n=[m['ceiling_peril_damage'],
       m['ceiling_no_damage'],
       m['gutter_peril_damage'],
       m['gutter_no_damage'],
       m['roof_peril_damage'],
       m['roof_no_damage'],
       m['stucco_peril_damage'],
       m['stucco_no_damage'],
       m['vent_peril_damage'],
       m['vent_no_damage']]
    return n

for i in list(image_profile.keys()):
    j = image_profile[i]
    t = '_'.join(j['image_name'].split('_')[2:4])
    j['image_name'] = t + '/' + j['image_name']
    j['features'] = new_vector(j['features'][:-9])
    k.append(j)

with open('dataset_list_'+ mode +'.pkl', 'wb') as handle:
    pickle.dump(k, handle, protocol=pickle.HIGHEST_PROTOCOL)

