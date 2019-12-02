#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 16:34:56 2019

@author: anup
"""

import torchvision.models as md
import torch.nn.functional as F
import torch.nn as nn


def vgg_model():
    vgg = md.vgg16(pretrained=True)
    num_features = vgg.classifier[6].in_features
    features = list(vgg.classifier.children())[:-1]
    features.extend([nn.Linear(num_features, 10)])
    vgg.classifier = nn.Sequential(*features)
    return vgg

def resnet18_model():
    resnet = md.resnet18(pretrained=True)
    num_features = resnet.fc.in_features
    features = list(resnet.fc.children())
    features.extend([nn.Linear(num_features, 10)])
    resnet.fc = nn.Sequential(*features)
    return resnet

def vgg_freeze_model():
    vgg = md.vgg16(pretrained=True)
    num_features = vgg.classifier[6].in_features
    features = list(vgg.classifier.children())[:-1]
    features.extend([nn.Linear(num_features, 10)])
    vgg.classifier = nn.Sequential(*features)
    for name_1, child in vgg.named_children():
        for name_2, params in child.named_parameters():
            if name_1 == 'features':
                params.requires_grad = False
    return vgg


def multi_label_vgg_model(training):
    vgg = md.vgg16(pretrained=True)
    num_features = vgg.classifier[6].in_features
    features = list(vgg.classifier.children())[:-1]
    if training:
        features.extend([nn.Linear(num_features, 2048),
                         nn.ReLU(inplace=True),
                         nn.Dropout(0.4),
                         nn.Linear(2048, 1024),
                         nn.ReLU(inplace=True),
                         nn.Dropout(0.4),
                         nn.Linear(1024, 16)])
    else:
        features.extend([nn.Linear(num_features, 2048),
                         nn.ReLU(),
                         nn.Linear(2048, 1024),
                         nn.ReLU(),
                         nn.Linear(1024, 16)])
    vgg.classifier = nn.Sequential(*features)
    return vgg

