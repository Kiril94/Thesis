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


def moveAllFilesinDir(srcDir, dstDir):
    """Moves all files in srcDir to dstDir."""
    if not(os.path.isdir(dstDir)):
        os.makedirs(dstDir) #Iterate over all the files in source directory
    for filePath in glob.iglob(srcDir + '\*'):
        shutil.move(filePath, dstDir);

def target_path(src_path, target_base_dir="G:/Cobra/Data"):
    """Turns source path (Y:/...) into target path, by defaul
    G:/Cobra/Data/..."""
    path_no_drive = os.path.splitdrive(dcm_file)[1]
    target_path = os.path.join(target_base_dir, path_no_drive)
    return target_path

def make_target_dir(src_dir, target_base_dir="G:/Cobra/Data"):
    """Takes source dir (Y:/...) and creates a path in the target dir
    "G:/Cobra/Data/..."""
    if not(os.path.exists(target_path(src_dir, target_base_dir=target_base_dir ))):
        os.makedirs(target_path(src_dir))
    
# In[tables directories]
src_dirs = os.listdir("Y:")
src_neg_dirs = sorted([f"{src_dirs}/{x}" for x \
                       in src_dirs if x.startswith('2019')])
dst_data_dir = "G:/Cobra/Data"

# In[move]


#for dir_ in healthy_dirs[:1]:
    #if not(os.path.exists(target_path(dir_))):
    #    os.mkdir(target_path(dir_))
    
    #patient_list = utils.list_subdir(dir_)
    
start = time.time()
for pat in patient_list[:2]:
    i = 0    
    for dcm_file in glob.iglob(f"{pat}/*/MR/*/*"):
        i+=1
        dst_file_dir = target_path(dcm_file)
        make_target_dir(os.path.split(dcm_file)[0])
        shutil.move(dcm_file, dst_file_dir);
        if i>5:
            break
stop = time.time()
print(stop-start)

start = time.time()
for pat in patient_list[:2]:
    i = 0
    for dcm_dir in glob.iglob(f"{pat}/*/MR/*"):
        i+=1
        make_target_dir(dcm_dir)
        dst_dir = target_path(dcm_dir)
        moveAllFilesinDir(dcm_dir, dst_dir)
        if i>5:
            break
stop = time.time()
print(stop-start)
        #moveAllFilesinDir(dcmDir, dstDir)
#path_parts = Path("Y:/2019_01\000d30ebf6b150c5b4bee2f199fbd210\8255fc9525809f4030ab3d022530a576\MR").parts
#new_path = Path(path_parts[0])/Path(path_parts[1])/Path(path_parts[2])
#os.path.split()        
#print(new_path)

#original = utils.list_subdir(table_dir)[0]
#target = f"{base_dir}/test.csv"
#shutil.move(original, target)