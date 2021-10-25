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
sys.path.append("D:/Thesis/Cobra")
from cobra.utilities import basic
from bs4 import BeautifulSoup
import gzip
import shutil
import nibabel as nib
nib.Nifti1Header.quaternion_threshold = -1e-06
import numpy as np

#%%
# In[Define directories]
script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parent.parent.parent.parent
data_folder = join(base_dir, "data")
task_folder = join(
    base_dir, data_folder,
    "nnunet/nnUNet_raw_data_base/nnUNet_raw_data/Task501_MICCAI")
tr_folder = join(task_folder, "imagesTr")
tr_files = basic.list_subdir(tr_folder)
tr_lbl_files = basic.list_subdir(join(task_folder, "labelsTr"))

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
                
for nii_file in tr_files:    
        if nii_file[-3:]=='nii':
            if os.path.exists(nii_file):
                os.remove(nii_file)

#%%
# In[Get the dict]
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
result_list = sorted(result_list)
result_list.insert(0, (0, 'background', '0 0 0'))
print(result_list[0])
print(b_label[0].find('Name').string)
with open(join(task_folder, 'original_labels.txt'), 'w') as f:
    for item in result_list:
        f.write(f"{item[0]}, {item[1]}, {item[2]}\n")
orig_labels_dic = {tuple_[0]:tuple_[1] for tuple_ in result_list}
with open(f"{task_folder}/original_labels_dic.txt", 'w') as f:
    print(orig_labels_dic, file=f)


#%%
# In[Convert labels to consecutive]
# The strategy is to fill missing labels by the last existing label,
# e.g. if 1 and 5 is missing we take 250 and make it 1, 249 is made 5
# List of relevant labels, all the rest is set to 0 (background)
labels_list = [46,4,49,51,50,52,23,36,57,55,
59,61,76,30,37,58,56,60,62,75,31,47,116,122,170,132,154,200,180,184,206,202,100,
138,166,102,172,104,136,146,178,112,118,120,124,140,152,186,142,162,164,190,204,
150,182,192,106,174,194,198,148,176,168,108,114,134,160,128,144,156,196,32,48,
117,123,171,133,155,201,181,185,207,203,101,139,167,103,173,105,137,147,179,
113,119,121,125,141,153,187,143,163,165,191,205,151,183,193,107,175,195,199,
149,177,169,109,115,135,161,129,145,157,197,44,45,11,35,38,40,39,41,71,72,73]
labels_list = sorted(labels_list)
trafo_dic = {x:(i+1) for i, x in enumerate(labels_list)}
def rename_keys(d, trafo_dic):
    d_new = d.copy()
    for item in trafo_dic.items():
        d_new[item[1]] = d_new.pop(item[0])
    return d_new
new_labels_dic = rename_keys(orig_labels_dic, trafo_dic)
new_labels_dic[0] = 'background'
with open(f"{task_folder}/new_labels_dic.txt", 'w') as f:
    print(new_labels_dic, file=f)

#%%
# In[Get trafo dict for images]
irr_labels = sorted(list(set(orig_labels_dic.keys()) - set(labels_list)))
for label in irr_labels: #map irrelevant labels to 0
    trafo_dic[label] = 0
print(trafo_dic)

with open(f"{task_folder}/trafo_dic.txt", 'w') as f:
    print(trafo_dic, file=f)
#inv_trafo_dic = {v: k for k, v in trafo_dic.items()}

#%%
# In[Transform images]
def transform_labels(im_path, trafo_dic):
    im = nib.load(im_path)
    arr = im.get_fdata().astype(np.int32)
    new_arr = np.vectorize(trafo_dic.get)(arr)
    nib.save(nib.Nifti1Image(new_arr, affine=im.affine), im_path)

transform = False
if transform:
    for im_path in tr_lbl_files:
        transform_labels(im_path, trafo_dic)
        print(f"transformed {im_path}")
#%%
# In[test transformed]
im = nib.load(tr_lbl_files[10])
arr = im.get_fdata().astype(np.int32)
print(np.unique(arr))

#%%
# In[wqw]
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




