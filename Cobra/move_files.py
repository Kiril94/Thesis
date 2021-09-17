# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 10:58:38 2021

@author: klein
"""

import shutil
import os
from pathlib import Path
from utilss import utils
import glob
import time


def target_path(src_path, target_base_dir="G:/CoBra/Data"):
    """Turns source path (Y:/...) into target path, by default
    G:/Cobra/Data/... . If target path does not exist, creates it. 
    We follow the structure month_dir/patient_id/scan_id/*.dcm"""
    path_no_drive = os.path.splitdrive(src_path)[1][1:] # first symbol is /
    split_path = Path(path_no_drive).parts
    main_path = split_path[0]
    patient_path = split_path[1]
    scan_path = split_path[4]
    target_path = os.path.join(
        target_base_dir, main_path, patient_path, scan_path)
    if not(os.path.exists(target_path)):
        os.makedirs(target_path)
    return target_path

    
# In[tables directories]
src_dirs = os.listdir("Y:")
src_neg_dirs = sorted([f"{src_dirs}/{x}" for x \
                       in src_dirs if x.startswith('2019')])
dst_data_dir = "G:/CoBra/Data"
print(Path('Y:/2019_01/000d30ebf6b150c5b4bee2f199fbd210/8255fc9525809f4030ab3d022530a576/MR/6cb3194cd9239cb24cc1a2a12506aa94').parts)
# In[move]


#for dir_ in healthy_dirs[:1]:
    #if not(os.path.exists(target_path(dir_))):
    #    os.mkdir(target_path(dir_))
    
    #patient_list = utils.list_subdir(dir_)
num_patients = 5
start = time.time()
for pat in patient_list[:num_patients]:
    i = 0    
    print('.')
    for dcm_file in glob.iglob(f"{pat}/*/MR/*/*"):
        i+=1
        dst_file_dir = target_path(
            Path(os.path.split(dcm_file)[0]), Path("G:/CoBra/Data/test1"))
        shutil.move(dcm_file, dst_file_dir);
        #if i>1:
        #    pass
stop = time.time()
print(f"The first method takes {(stop-start)/num_patients} per patient")

