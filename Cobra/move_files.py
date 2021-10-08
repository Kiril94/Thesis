# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 10:58:38 2021

@author: klein
"""

import shutil
import os
from os.path import join
from pathlib import Path
from utilss import utils
import glob
import time
import numpy as np
import pandas as pd


print("We will download only dwi, swi, flair, t1, t2, t2*")
print("Start with smallest group of patients (1104) dwi, flair, t2*, t1, mostly negative patients,")

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
script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parent
src_dirs = os.listdir("Y:")
src_neg_dirs = sorted([f"{src_dirs}/{x}" for x \
                       in src_dirs if x.startswith('2019')])
dst_data_dir = "G:/CoBra/Data"
download_pat_path = join(base_dir, "share/Cerebriu/download_patients")

# In[Get relevant patients and volumes]
print(os.listdir(download_pat_path))
rel_seq = ['dwi', 'swi', 't1', 't2', 't2s', 'flair']
dwi_flair_t2s_t1_list = np.loadtxt(join(download_pat_path, "dwi_flair_t2s_t1.txt"),
                                   dtype='str')
df_patients = pd.read_csv(join(base_dir, "share/pred_seq.csv"))

df_patients_0 = df_patients[df_patients['PatientID'].isin(dwi_flair_t2s_t1_list)]
df_patients_0 = df_patients_0[df_patients_0['Sequence'].isin(rel_seq)]
df_patients_0 = df_patients_0.sort_values('PatientID')

volume_dir_df = pd.read_csv(join(base_dir, 'tables', 'sid_directories.csv'))
volume_dir_dic = pd.Series(
    volume_dir_df.Directory.values, index=volume_dir_df.SeriesInstanceUID).to_dict()


# In[move]
# 1st patient was already written
# last patient: 034eb2b7527b2db0857386daafd05d41
for pat in df_patients_0.PatientID.unique()[1:]:
    start = time.time()
    print(f"patient: {pat}:", end=' ')
    volumes = df_patients_0[df_patients_0.PatientID==pat]['SeriesInstanceUID']
    print(f"{len(volumes)} volumes")
    for volume in volumes:
        volume_dir = volume_dir_dic[volume]
        for dcm_file in glob.iglob(f"Y:/{volume_dir}/*"):
            dst_file_dir = target_path(
                Path(os.path.split(dcm_file)[0]), Path("G:/CoBra/Data/dcm"))
            try:
                shutil.move(dcm_file, dst_file_dir);
            except:
                print(f"Destination path {dst_file_dir} already exists")
        print("|",  end='')
    stop = time.time()
    print(f" {(stop-start)/60} mins")



