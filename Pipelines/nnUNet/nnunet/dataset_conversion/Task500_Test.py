#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 16:54:47 2021

@author: kiril
"""
#%%
# In[Import]
import utils
from pathlib import Path
from os.path import join
from os.path import split
import os
import sys
import json
sys.path.append("D:/Thesis/Cobra")
from cobra.utilities import basic
from bs4 import BeautifulSoup
import gzip
import shutil

#%%
# In[Define directories]
script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parent.parent.parent.parent
data_folder = join(base_dir, "data")
task_folder = join(
    base_dir, data_folder,
    "nnUNet_raw_data_base/nnUNet_raw_data/Task500_Test")
tr_folder = join(task_folder, "imagesTr")
ts_folder = join(task_folder, "imagesTs")
tr_files = basic.list_subdir(tr_folder)
ts_files = basic.list_subdir(ts_folder)
tr_lbl_files = basic.list_subdir(join(task_folder, "labelsTr"))
ts_lbl_files = basic.list_subdir(join(task_folder, "labelsTs"))

#%%
# In[Rename files]
# The last index should correspond to the modality
rename=False

for file in ts_lbl_files:
    dir_, name = split(file)
    #name = "MICCAI_"+name[1:5]+'.nii'
    print(name)
    name = name[:-8] + name[-7:]
    print(name)
    if rename:
        target_file = join(
            dir_, name)
        os.rename(file, target_file)
#%%
# In[gzip files]
# test
zip_files=False
if zip_files:
    for nii_file in ts_lbl_files:    
        with open(nii_file, 'rb') as f_in:
            with gzip.open(nii_file+'.gz', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
                
for nii_file in ts_lbl_files:    
        if nii_file[-3:]=='nii':
            if os.path.exists(nii_file):
                os.remove(nii_file)
#%%
# In[Convert labels to nnunet]
def convert_to_nnunet():
    pass

def convert_back_to_miccai():
    pass

#%%
# In[Get labels]
xml_dir = join(task_folder, "labels_dict.xml")
with open(xml_dir, 'r') as f:
    data = f.read()
 
Bs_data = BeautifulSoup(data, "xml")
result_list = []
b_label = Bs_data.find_all('Label')
for b in b_label:
    number = b.find('Number').string
    if len(number)==1:
        number = '00'+number
    elif len(number)==2:
        number = '0'+number
    else: 
        pass
    name = b.find('Name').string
    color = b.find('RGBColor').string
    result_list.append((number, name, color))

print(result_list[0])
#print(dir(b_label[0].find('Name')))
print(b_label[0].find('Name').string)
with open(join(task_folder, 'labels.txt'), 'w') as f:
    for item in result_list:
        f.write(f"{item[0]}, {item[1]}, {item[2]}\n")
#print(result)
#print(b_label.getitem())
# %%
# In[Create Labels]
labels_dic = {tuple_[0]:tuple_[1] for tuple_ in result_list}
labels_dic['000'] = 'background'
labels_dic = dict(sorted(labels_dic.items(), key=lambda t:t[0]))
with open(join(task_folder, 'labels_dict.txt'), 'w') as f:
    print(labels_dic, file=f)

#%%
# In[generate dataset json]
utils.generate_dataset_json(join(task_folder, 'dataset.json'), 
                            tr_folder, 
                            ts_folder, 
                            modalities=('T1',),
                            labels=labels_dic, 
                            dataset_name="Task500_Test", 
                           )
#%%
# In[Compress tar files]
import tarfile
with tarfile.open(join(data_folder, 'share',"Task500_Test.tar.gz"), "w:gz") as tar:
    tar.add(task_folder, arcname="Task500_Test")

#%%
# In[To decompress the file run]

with tarfile.open('test2.tar.gz') as file:
    file.extractall('test_extract')




