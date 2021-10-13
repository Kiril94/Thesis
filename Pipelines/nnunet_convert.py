#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 16:54:47 2021

@author: kiril
"""
from nnUNet.nnunet.dataset_conversion import utils
from pathlib import Path
from os.path import join
from os.path import split
import os
import sys
sys.path.append("D:/Thesis/Cobra")
from cobra.utilities import basic
from bs4 import BeautifulSoup
import gzip
import shutil

# In[Define directories]
script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parent
data_folder = join(base_dir, "data")
task_folder = join(
    base_dir, data_folder,
    "nnUNet_raw_data_base/nnUNet_raw_data/Task500_Test")

train_folder = join(task_folder, "imagesTr")
test_folder = join(task_folder, "imagesTs")
train_labels_folder = join(task_folder, "labelsTr")
train_files = basic.list_subdir(train_folder)

train_files = basic.list_subdir(train_folder)
test_files = basic.list_subdir(test_folder)
train_labels_files = basic.list_subdir(train_labels_folder)

# In[Rename files]
# The last index should correspond to the modality
rename=False
if rename:
    for file in train_labels_files:
        dir_, name = split(file)
        name = "MICCAI_"+name[1:5]+'0000.nii'
        target_file = join(
            dir_, name)
        os.rename(file, target_file)
# In[gzip files]
# test
zip_files=False
if zip_files:
    for nii_file in test_files:    
        with open(nii_file, 'rb') as f_in:
            with gzip.open(nii_file+'.gz', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
                
for nii_file in train_labels_files:    
        if nii_file[-3:]=='nii':
            if os.path.exists(nii_file):
                os.remove(nii_file)
# In[test]
test = 'a_1128_3_0000.nii'
print(test[:2]+test[3:7]+test[8:])

# In[Get labels]
xml_dir = join(task_folder, "labels_dict.xml")
with open(xml_dir, 'r') as f:
    data = f.read()
 
Bs_data = BeautifulSoup(data, "xml")
result_list = []
b_label = Bs_data.find_all('Label')
for b in b_label:
    number = int(b.find('Number').string)
    name = b.find('Name').string
    color = b.find('RGBColor').string
    result_list.append((number, name, color))

print(result_list[0])
#print(dir(b_label[0].find('Name')))
print(b_label[0].find('Name').string)
#print(result)
#print(b_label.getitem())
# In[Create Labels]
labels_dic = {tuple_[0]:tuple_[1] for tuple_ in result_list}
with open(join(task_folder, 'labels_dict.txt'), 'w') as f:
    print(labels_dic, file=f)

# In[generate dataset json]
utils.generate_dataset_json(join(task_folder, 'dataset.json'), 
                            train_folder, 
                            test_folder, 
                            modalities=('T1',),
                            labels=labels_dic, 
                            dataset_name="Task500_Test", 
                           )

# In[Compress tar files]
import tarfile
with tarfile.open(join(data_folder, 'share',"Task500_Test.tar.gz"), "w:gz") as tar:
    tar.add(task_folder, arcname="Task500_Test")


# In[To decompress the file run]

with tarfile.open('test2.tar.gz') as file:
    file.extractall('test_extract')




