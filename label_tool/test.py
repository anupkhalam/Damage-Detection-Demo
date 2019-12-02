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
