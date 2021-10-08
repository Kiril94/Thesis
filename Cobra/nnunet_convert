#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 16:54:47 2021

@author: kiril
"""
from nnUNet.nnunet.dataset_conversion import utils
from nnUNet.nnunet.preprocessing.utils import xml_parser
from pathlib import Path
from os.path import join
import os
from utilss import basic
from xml.dom import minidom
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup

# In[Define directories]
script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parent
data_folder = join(base_dir, "data/data_folder")
task_folder = join(
    base_dir, data_folder,
    "nnUNet_raw_data_base/nnUNet_raw_data/Task500_Test")

train_folder = join(task_folder, "imagesTr")
test_folder = join(task_folder, "imagesTs")
train_labels_folder = join(task_folder, "labelsTr")
train_files = basic.list_subdir(train_folder)

# In[Rename files]
# The last index should correspond to the modality
# we rename all files to _0000.nii
test_files = basic.list_subdir(test_folder)
train_labels_files = basic.list_subdir(train_labels_folder)
#for file in train_labels_files:
#    os.rename(file, file[:-8]+".nii")

# In[Get labels]
xml_dir = join(data_folder, "labels_dict.xml")


with open(xml_dir, 'r') as f:
    data = f.read()
 
# Passing the stored data inside
# the beautifulsoup parser, storing
# the returned object
Bs_data = BeautifulSoup(data, "xml")
 
# Finding all instances of tag
# `unique`
result_list = []
b_label = Bs_data.find_all('Label')
for b in b_label:
    number = int(b.find('Number').string)
    name = b.find('Name').string
    color = b.find('RGBColor').string
    result_list.append((number, name, color))

print(result_list[0])
print(dir(b_label[0].find('Name')))
print(b_label[0].find('Name').string)

#print(result)
#print(b_label.getitem())
# In[]
utils.generate_dataset_json(join(task_folder, 'dataset.json'), 
                            train_folder, 
                            test_folder, 
                            ('T1'),
                          labels={0: 'background', 1: 'street'}, 
                          dataset_name="Task500_Test", 
                          license='hands off!')
