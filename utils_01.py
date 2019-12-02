#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 16:43:01 2019

@author: anup
"""

# =============================================================================
# Importing Libraries
# =============================================================================
import os
from PIL import Image
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt; plt.rcdefaults()
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# =============================================================================
# Defining Parameters
# =============================================================================
TRANSFORM_IMG = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225] )
    ])

# =============================================================================
# Defining Class labels
# =============================================================================
elevation_items = ['Ceiling',
                   'Gutter',
                   'Roof',
                   'Stucco',
                   'Vent',
                   'Downspout',
                   'Door',
                   'Windows']

elevation_rows = ['_'.join(i.lower().split()) for i in elevation_items]

damage_status_pre = ['No Damage','Peril Damage']
damage_status = ['_'.join(i.lower().split()) for i in damage_status_pre]

dataset_rows = []
dataset_rows_dict = {}
for h, i in enumerate(elevation_rows):
    for k, j in enumerate(damage_status):
        dataset_rows.append(i + '_'+ j)
        dataset_rows_dict[i + '_'+ j] = [elevation_items[h], damage_status_pre[k]]



# =============================================================================
# Image Processing
# =============================================================================
def image_loader(image_name):
    """Load the images for prediction"""
    image = Image.open(image_name)
    image = TRANSFORM_IMG(image).float()
    image = image.unsqueeze(0)
    return image

#==============================================================================
# Run the MODEL for single category with two sub categories
#==============================================================================
def test_images(model,
                image_list,
                device):

    """Routine to test single category"""

    def_counter = 0
    nondef_counter = 0
    for file in image_list:
        image = image_loader(file)
        output = int(torch.argmax(model(image.to(device))))
        if output == 0:
            print(file.split('/')[-1], ": ", "DEFECTIVE")
            def_counter += 1
        elif output == 1:
            print(file.split('/')[-1], ": ", "NON-DEFECTIVE")
            nondef_counter += 1
        else:
            print("ERROR")
    return [def_counter, nondef_counter]

#==============================================================================
# Run the MODEL for all categories and subcategories
#==============================================================================
def test_image_api(model, image_list, device):
    model.eval()
    class_label_dict = load_dict('class_label_dict')
    for file in image_list:
        image = image_loader(file)
        output = int(torch.argmax(model(image.to(device))))
    label = class_label_dict[output]
    return label.split('_')

SM = nn.Softmax(dim=1)
def test_image_api_multi(model, image_list, device):
    model.eval()
    class_label_dict = load_dict('class_label_dict')
    for file in image_list:
        image = image_loader(file)
        flt_output = model(image.to(device))
        int_output = SM(flt_output)
        int_output = int_output.detach()
        int_output = torch.flatten(int_output)
        int_output.tolist()
        int_output = [float(i) for i in int_output]
        output_index = sorted(range(len(int_output)), key=lambda k: int_output[k])[::-1]
        print("#######", output_index)
        cat_list = []
        stat_list = []
        for index_ in output_index:
            if int_output[index_] > 0.1:
                item = class_label_dict[index_]
                cat_item = item.split('_')[0]
                stat_item = item.split('_')[1]
                if cat_item not in cat_list:
                    cat_list.append(cat_item)
                    stat_list.append(stat_item)
        result_ = [(cat_list[i],stat_list[i]) for i in range(len(cat_list))]
        
    return result_

#==============================================================================
# Get details of the training images
#==============================================================================
def get_train_details(path,
                      class_list):

    """Fetch details of training images"""

    counter_d = 0
    counter_nd = 0
    train_image_list = []
    for label in class_list:
        label_d = path + label + '_dam/'
        label_nd = path + label + '_nodam/'
        print("="*10, label.upper(), "="*(40-len(label)-2))
        label_d_img = glob.glob(label_d + '*.jpg')
        train_image_list.extend(label_d_img)
        counter_d += len(label_d_img)
        print("# of training images for damaged {} : {}".format(label, len(label_d_img)))
        label_nd_img = glob.glob(label_nd + '*.jpg')
        train_image_list.extend(label_nd_img)
        counter_nd += len(label_nd_img)
        print("# of training images for undamaged {} : {}".format(label, len(label_nd_img)))
        print("="*50)
        print(" "*50)
    print("Total # of training images : {}".format(counter_d + counter_nd))
    return train_image_list

#==============================================================================
# Get details of the test images
#==============================================================================
def get_test_details(path,
                     class_list):

    """Fetch details of the test images"""

    counter_d = 0
    counter_nd = 0
    for label in class_list:
        label_d = path + label + '_dam/'
        label_nd = path + label + '_nodam/'
        print("="*10, label.upper(), "="*(40-len(label)-2))
        label_d_img = glob.glob(label_d + '*.jpg')
        counter_d += len(label_d_img)
        print("# of test images for damaged {} : {}".format(label, len(label_d_img)))
        label_nd_img = glob.glob(label_nd + '*.jpg')
        counter_nd += len(label_nd_img)
        print("# of test images for undamaged {} : {}".format(label, len(label_nd_img)))
        print("="*50)
        print(" "*50)
    print("Total # of test images : {}".format(counter_d + counter_nd))

#==============================================================================
# Predict the status of the image
#==============================================================================
def test_damage_images(model,
                       device,
                       path,
                       class_list):

    """Predict the status of the image"""

    model = model.eval()
    result_list = []
    class_label_dict = load_dict('class_label_dict')
    for label in class_list:
        label_d = path + label + '_dam/*.jpg'
        label_d_img = glob.glob(label_d)
        for file in label_d_img:
            image = image_loader(file)
            output = int(torch.argmax(model(image.to(device))))
            result_list.append((file,
                                class_label_dict['_'.join(file.split('/')[-1][12:].split('_')[:2])],
                                output))
          
        label_nd = path + label + '_nodam/*.jpg'
        label_nd_img = glob.glob(label_nd)
        for file in label_nd_img:
            image = image_loader(file)
            output = int(torch.argmax(model(image.to(device))))
            result_list.append((file,
                                class_label_dict['_'.join(file.split('/')[-1][12:].split('_')[:2])],
                                output))
    return result_list
          
#==============================================================================
# Save the label id dictionary
#==============================================================================

def save_dict(obj,
              name):

    """Save the label id dictionary"""

    with open('label/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

#==============================================================================
# Load the label id dictionary
#==============================================================================
def load_dict(name):

    """Load the label id dictionary"""

    with open('label/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

#==============================================================================
# Get the accuracy of categorizing
#==============================================================================
def categorizing_accuracy(result_list):

    """Get category based accuracy"""

    class_label_dict = load_dict('class_label_dict')
    counter = 0
    counter_correct = 0
    for _, s_index, o_index in result_list:
        counter += 1
        if class_label_dict[s_index].split('_')[0] == class_label_dict[o_index].split('_')[0]:
            counter_correct += 1
    if counter > 0:
        category_result = [counter_correct,
                       counter - counter_correct,
                       round((counter_correct/counter)*100,2)]
        print("Number of correct predictions : {}".format(category_result[0]))
        print("Number of wrong predictions : {}".format(category_result[1]))
        print("="*50)
        print("Category level accuracy : {} %".format(category_result[2]))
        print("="*50)
        return category_result
    else:
        category_result = [0,0,0]
        print("Number of correct predictions : {}".format(category_result[0]))
        print("Number of wrong predictions : {}".format(category_result[1]))
        print("="*50)
        print("Category level accuracy : {} %".format(category_result[2]))
        print("="*50)
        return category_result

#==============================================================================
# Get the subcategory based accuracy
#==============================================================================
def overall_accuracy(result_list):

    """Accuracy based on sub-category"""

    counter = 0
    counter_correct = 0
    for _, s_index, o_index in result_list:
        counter += 1
        if s_index == o_index:
            counter_correct += 1
    if counter > 0:
        overall_result = [counter_correct,
                          counter - counter_correct,
                          round((counter_correct/counter)*100,2)]
        print("Number of correct predictions : {}".format(overall_result[0]))
        print("Number of wrong predictions : {}".format(overall_result[1]))
        print("="*50)
        print("Sub-category level accuracy : {} %".format(overall_result[2]))
        print("="*50)
        return overall_result
    else:
        overall_result = [0, 0, 0]
        print("Number of correct predictions : {}".format(overall_result[0]))
        print("Number of wrong predictions : {}".format(overall_result[1]))
        print("="*50)
        print("Sub-category level accuracy : {} %".format(overall_result[2]))
        print("="*50)
        return overall_result

#==============================================================================
# Get the details of test images for further processing 
#==============================================================================
def process_result(result_list):

    """Gather truth and prediction for test images"""

    class_label_dict = load_dict('class_label_dict')
    result_dict = {}
    result_dict['path'] = []
    result_dict['truth'] = []
    result_dict['predicted'] = []
    result_incorrect_cat_dict = {}
    result_incorrect_cat_dict['path'] = []
    result_incorrect_cat_dict['truth'] = []
    result_incorrect_cat_dict['predicted'] = []
    result_incorrect_sub_cat_dict = {}
    result_incorrect_sub_cat_dict['path'] = []
    result_incorrect_sub_cat_dict['truth'] = []
    result_incorrect_sub_cat_dict['predicted'] = []

    for file, s_index, o_index in result_list:
        result_dict['path'].append(file)
        result_dict['truth'].append(s_index)
        result_dict['predicted'].append(o_index)
        if s_index != o_index:
            result_incorrect_sub_cat_dict['path'].append(file)
            result_incorrect_sub_cat_dict['truth'].append(s_index)
            result_incorrect_sub_cat_dict['predicted'].append(o_index)
        if class_label_dict[s_index].split('_')[0] != class_label_dict[o_index].split('_')[0]:
            result_incorrect_cat_dict['path'].append(file)
            result_incorrect_cat_dict['truth'].append(s_index)
            result_incorrect_cat_dict['predicted'].append(o_index)


    result_dataset = pd.DataFrame(result_dict)
    result_dataset['path'] = result_dataset['path'].apply(lambda x:x.split('/')[-1])
    result_dataset['truth'] = result_dataset['truth'].apply(lambda x:class_label_dict[x])
    result_dataset['predicted'] = result_dataset['predicted'].apply(lambda x:class_label_dict[x])

    result_incorrect_dataset = pd.DataFrame(result_incorrect_sub_cat_dict)
    result_incorrect_dataset['path'] = result_incorrect_dataset['path'].apply(lambda x:x.split('/')[-1])
    result_incorrect_dataset['truth'] = result_incorrect_dataset['truth'].apply(lambda x:class_label_dict[x])
    result_incorrect_dataset['predicted'] = result_incorrect_dataset['predicted'].apply(lambda x:class_label_dict[x])
#
    result_incorrect_dataset_cat = pd.DataFrame(result_incorrect_cat_dict)
    result_incorrect_dataset_cat['path'] = result_incorrect_dataset_cat['path'].apply(lambda x:x.split('/')[-1])
    result_incorrect_dataset_cat['truth'] = result_incorrect_dataset_cat['truth'].apply(lambda x:class_label_dict[x])
    result_incorrect_dataset_cat['predicted'] = result_incorrect_dataset_cat['predicted'].apply(lambda x:class_label_dict[x])
#
    result_dataset.to_csv('result/result.csv', index=False)
    result_incorrect_dataset.to_csv('result/wrong_predictions_sub_cat.csv', index=False)
    result_incorrect_dataset_cat.to_csv('result/wrong_predictions_cat.csv', index=False)
#
    return result_dict, result_incorrect_cat_dict, result_incorrect_sub_cat_dict

#==============================================================================
# Display incorrect images
#==============================================================================
def get_images(image_index,
               incorrect):

    """Display sample wrong predictions"""

    class_label_dict = load_dict('class_label_dict')
    for i in image_index:
        img = mpimg.imread(incorrect['path'][i])
        imgplot = plt.imshow(img)
        plt.show()
        print("True Category : {}".format(class_label_dict[incorrect['truth'][i]]))
        print("Predicted Category : {}".format(class_label_dict[incorrect['predicted'][i]]))
        print("="*50)
        print(" "*50)

#==============================================================================
# Prepare for accuracy display
#==============================================================================
def get_label_count(result):

    """Get the count of categories"""

    class_label_dict = load_dict('class_label_dict')
    c_sub_labels = list(class_label_dict.keys())
    c_sub_labels = [i for i in c_sub_labels if type(i) != int]
    c_labels = list(set([i.split('_')[0] for i in c_sub_labels]))
    type_list = [k.split('/')[-1].split('.')[0][12:] for k in result]
    cat_count_list=[k.split('_')[0] for k in type_list]
    type_dict = {x:cat_count_list.count(x) for x in cat_count_list}
    for label in c_labels:
        if label not in list(type_dict.keys()):
            type_dict[label] = 0
    
    ovr_cat_count_list=['_'.join(k.split('_')[0:2]) for k in type_list]
    ovr_type_dict = {x:ovr_cat_count_list.count(x) for x in ovr_cat_count_list}
    for sublabel in c_sub_labels:
        if sublabel not in list(ovr_type_dict.keys()):
            ovr_type_dict[sublabel] = 0
    return type_dict, ovr_type_dict


#==============================================================================
# Display bar chart
#==============================================================================
def display_res_barchart(display_dict,
                         y_label,
                         title_,
                         margin_):

    """Display bar chart"""

    objects = list(display_dict.keys())
    object_value = list(display_dict.values())
    y_pos = np.arange(len(objects))
    plt.bar(y_pos, object_value, align='center', alpha=0.5)
    plt.xticks(y_pos, objects,rotation=45, ha='right')
    for i, v in enumerate(object_value):
        plt.text(y_pos[i] - 0.2, v + 0.5, str(v) + " %", rotation=45)
    plt.ylabel(y_label)
    plt.title(title_)
    plt.margins(margin_)
    plt.show()

#==============================================================================
# Display the ctaegory count
#==============================================================================
def display_cat_count_barchart(train_cat_type_dict,
                               train_image_length):

    """Display the ctaegories count"""

    train_cat_type_per_dict = {}
    for k in list(train_cat_type_dict.keys()):
        train_cat_type_per_dict[k] = round((train_cat_type_dict[k]/(train_image_length))*100, 2)

    fig = plt.figure(figsize=(10,8))
    fig.suptitle("Category Distribution of Training Images")
    plt.subplot(1, 2, 1)
    objects = list(train_cat_type_dict.keys())
    object_value = list(train_cat_type_dict.values())
    y_pos = np.arange(len(objects))
    plt.bar(y_pos, object_value, align='center', alpha=0.5)
    plt.xticks(y_pos, objects,rotation=45, ha='right')
    for i, v in enumerate(object_value):
        plt.text(y_pos[i] - 0.2, v + 0.5, str(v), rotation=45)
    plt.ylabel("Number of Images")

    plt.margins(0.1)
    plt.subplot(1, 2, 2)
    objects = list(train_cat_type_per_dict.keys())
    object_value = list(train_cat_type_per_dict.values())
    y_pos = np.arange(len(objects))
    plt.bar(y_pos, object_value, align='center', alpha=0.5)
    plt.xticks(y_pos, objects,rotation=45, ha='right')
    for i, v in enumerate(object_value):
        plt.text(y_pos[i] - 0.2, v + 0.5, str(v) + " %", rotation=45)
    plt.ylabel("% within Training Images")
    plt.margins(0.1)
    plt.show()

#==============================================================================
# Display sub category count
#==============================================================================
def display_subcat_count_barchart(train_ovr_cat_type_dict,
                                  train_image_length):

    """Display sub ctaegory count"""

    train_subcat_type_per_dict = {}
    for k in list(train_ovr_cat_type_dict.keys()):
        train_subcat_type_per_dict[k] = round((train_ovr_cat_type_dict[k]/(train_image_length))*100, 2)

    fig = plt.figure(figsize=(10, 8))
    fig.suptitle("Sub Category Distribution of Training Images")
    plt.subplot(1, 2, 1)
    objects = list(train_ovr_cat_type_dict.keys())
    object_value = list(train_ovr_cat_type_dict.values())
    y_pos = np.arange(len(objects))
    plt.bar(y_pos, object_value, align='center', alpha=0.5)
    plt.xticks(y_pos, objects,rotation=45, ha='right')
    for i, v in enumerate(object_value):
        plt.text(y_pos[i] - 0.2, v + 0.5, str(v),
                 rotation=45,
                 fontsize=8)
    plt.ylabel("Number of Training Images")
    
    plt.margins(0.075)
    plt.subplot(1, 2, 2)
    objects = list(train_subcat_type_per_dict.keys())
    object_value = list(train_subcat_type_per_dict.values())
    y_pos = np.arange(len(objects))
    plt.bar(y_pos, object_value, align='center', alpha=0.5)
    plt.xticks(y_pos, objects,rotation=45, ha='right')
    for i, v in enumerate(object_value):
        plt.text(y_pos[i] - 0.2, v + 0.5, str(v) + " %",
                 rotation=45,
                 fontsize=8)
    plt.ylabel("% within Training Images")
    plt.margins(0.075)
    plt.show()
    
# =============================================================================
# Dataset Class
# =============================================================================
class MultiLabelDataset(Dataset):
    def __init__(self, img_file, root_dir, transform=None):
        with open(img_file, 'rb') as handle:
            self.img_dataset = pickle.load(handle)
        self.root_dir = root_dir
        label_list = dataset_rows
        self.class_to_idx = {k: v for v, k in enumerate(label_list)}
        self.transform = transform

    def __len__(self):
        return len(self.img_dataset)

    def __getitem__(self, id_):
        if torch.is_tensor(id_):
            id_ = id_.tolist()
        img_name = os.path.join(self.root_dir,
                                self.img_dataset[id_]['image_name'])
        image = Image.open(img_name)
        comp_vector = self.img_dataset[id_]['features']
        comp_vector = np.array(comp_vector)
        sample = {'image': image,
                  'comp_vector': comp_vector}

        if self.transform:
            sample = self.transform(sample)

        return sample

# =============================================================================
# Image Rescale Class
# =============================================================================
class ReSize():
    
    def __init__(self, resize_value):
        assert isinstance(resize_value, tuple)
        self.resize_value = resize_value
    
    def __call__(self, sample):
        assert isinstance (sample, dict)
        resize_object = transforms.Resize(self.resize_value)
        sample['image'] = resize_object(sample['image'])
        return sample

class ImCrop():

    def __init__(self, crop_value):
        assert isinstance(crop_value, int)
        self.crop_value = crop_value

    def __call__(self, sample):
        assert isinstance(sample, dict)
        crop_object = transforms.CenterCrop(self.crop_value)
        sample['image'] = crop_object(sample['image'])
        return sample

class TensorConv():
    def __call__(self, sample):
        assert isinstance(sample, dict)
        tensor_conv_object = transforms.ToTensor()
        normalize_object = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
        sample['image'] = tensor_conv_object(sample['image'])
        sample['comp_vector'] = torch.from_numpy(sample['comp_vector']).float()
        sample['image'] = normalize_object(sample['image'])
        return sample

# =============================================================================
# Test Multi Label API
# =============================================================================
def test_mlable_api(model, image_list, device):
    model.eval()
    class_label_dict = load_dict('class_label_dict')
    for file in image_list: # Expecting only one file in image_list
        image = image_loader(file)
        output = model(image.to(device))
        output_cpu = output.squeeze(0)

        arg_max = torch.argmax(output_cpu)
        output_cpu[arg_max] = 1
        output_cpu[output_cpu < 0] = 0
        output_cpu[output_cpu > 0] = 1
        det_comp = torch.nonzero(output_cpu.squeeze(0)==1.).view(1,-1)[0].tolist()
#        if not det_comp:
#            output_cpu[max(output_cpu)] = 1
#            output_cpu[output_cpu < 0] = 0
#            det_comp = torch.nonzero(output_cpu.squeeze(0)==1.).view(1,-1)[0].tolist()
        model_output = [(class_label_dict[i].split('_')[0].upper(), class_label_dict[i].split('_')[1].upper()) \
                        for i in det_comp]
    item_list = []
    new_model_output = []
    if model_output:
        for item in model_output:
            if item[0] not in item_list:
                new_model_output.append(item)
                item_list.append(item[0])
    return new_model_output

# =============================================================================
 





