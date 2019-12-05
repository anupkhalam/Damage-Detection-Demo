#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 10:38:14 2019

@author: anup
"""
#import os
#import cv2
#import numpy as np
#	
## Get the list of all files in directory tree at given path
#train_list = list()
#train_2_list = list()
#train_folder = '/home/anup/002_work/007_dam_det/05_label_tool/dataset_single/pdf_out/Door'
#for (dirpath, dirnames, filenames) in os.walk(train_folder):
#    train_list += [os.path.join(dirpath, file) for file in filenames if file.endswith('.jpg')]
#for (dirpath, dirnames, filenames) in os.walk(train_folder):
#    train_2_list += [os.path.join(dirpath, file) for file in filenames if file.endswith('.jpg')]
#
#
#
##test_list=[]
##test_folder = '/home/anup/work/multilabel/dataset_single/test'
##for (dirpath, dirnames, filenames) in os.walk(test_folder):
##    test_list += [os.path.join(dirpath, file) for file in filenames if file.endswith('.jpg')]
##orginal = '/home/anup/002_work/007_dam_det/05_label_tool/dataset_single/pdf_out/Vent/11455_14_3.jpg'
#original_ = cv2.imread(orginal)
#
#duplicate_counter=0
#duplicates=[]
#for index, duplicate in enumerate(train_list):
#    print("{}/{}".format(index, len(train_list)))
#    duplicate_ = cv2.imread(duplicate)
#
#    # 1) Check if 2 images are equals
#    if original_.shape == duplicate_.shape:
#        difference = cv2.subtract(original_, duplicate_)
#        b, g, r = cv2.split(difference)
#        if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
#            duplicate_counter += 1
#            duplicates.append((orginal, duplicate))
#            print("The images are completely Equal")
#    else:
#        continue
#
#from PIL import Image
#k=Image.open('/home/anup/002_work/007_dam_det/05_label_tool/dataset_single/pdf_out/Vent/11027_50_3.jpg')
#k.show()
#import os
#for i in duplicates:
#    os.remove(i[1])
#os.remove(orginal)
#
#
#k=[]
#l=[]
#k = [cv2.imread(i) for i in train_list]
#
#for i, j in enumerate(k):
#    if j in c:
#        continue
#    else:
#        l.append(train_list[i])
#        c.append(j)
#    
#len(l)
#len(k)


import os
import cv2
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
# Get the list of all files in directory tree at given path
train_list = list()
train_2_list = list()
train_folder = '/home/anup/work/multilabel/label_tool/dataset_single/new_data'
for (dirpath, dirnames, filenames) in os.walk(train_folder):
    train_list += [os.path.join(dirpath, file) for file in filenames if file.endswith('.jpg')]

counter=0
c=[]
t1 = datetime.now()
for i, j in enumerate(train_list):
    counter+=1
    print(str(counter) + ' / ' + str(len(train_list)))
    print("Time: ", (datetime.now() - t1).total_seconds())
    t1 = datetime.now()
    t = cv2.imread(j)
    for k in range(i+1, len(train_list)):
        u = cv2.imread(train_list[k])
        if t.size != u.size:
            continue
        else:
            try:
                difference = cv2.subtract(t, u)
                b, g, r = cv2.split(difference)
                if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
                    print("*************")
                    print(j, '\n', train_list[k])
                    c.append((j, train_list[k]))
#                t = t.reshape(-1, t.shape[1])
#                u = u.reshape(-1, u.shape[1])
#                cos_sim = spatial.distance.cosine(t.flatten(), u.flatten())
#                cos_sim = cv2.matchTemplate(t, u, cv2.TM_SQDIFF_NORMED)
#                print("C S: ", cos_sim)
#                if cos_sim < -0.99:
#                    print("*************")
#                    print(j, '\n', train_list[k])
#                    c.append((j, train_list[k]))
            except:
                print("error")
                continue
        
#for i in c:
#    os.remove(i[1])
    

import json
import glob
with open('dataset.json', 'rb') as f:
    k = json.load(f)

a = list(k.keys())
b = glob.glob('/home/anup/work/multilabel/label_tool/dataset_single/new_data/*.jpg')
b = [i.split('/')[-1] for i in b]

t={}
for i in a:
    if i in b:
        t[i] = k[i]
        print (i)

with open('dataset_v1.json', 'w') as w:
    json.dump(t, w)







