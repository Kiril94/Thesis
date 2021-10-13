# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 10:58:38 2021

@author: klein
"""

import shutil
import os
from os.path import join
from pathlib import Path
import glob
import time
import numpy as np
import pandas as pd
from utilities.utils import target_path

print("We will download only dwi, swi, flair, t1, t2, t2*")
print("Start with smallest group of patients (1104) dwi, flair, t2*, t1, mostly negative patients,")

    
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


# In[get index of last patient]
# 1st patient was already written
# last patient: 0385ef30676c4602159171edac0cc2d6
patient_list = df_patients_0.PatientID.unique()
last_patient = "142f12fa4858d6cc5093b824fe9510db"
last_patient_idx = np.where(patient_list==last_patient)[0][0]
print(patient_list[last_patient_idx:])
# In[move]
for pat in patient_list[last_patient_idx:]:
    start = time.time()
    print(f"patient: {pat}:", end=' ')
    volumes = df_patients_0[df_patients_0.PatientID==pat]['SeriesInstanceUID']
    print(f"{len(volumes)} volumes")
    for volume in volumes:
        volume_dir = volume_dir_dic[volume]
        counter = 0
        for dcm_file in glob.iglob(f"Y:/{volume_dir}/*"):
            counter+=1
            dst_file_dir = target_path(
                Path(os.path.split(dcm_file)[0]), Path("G:/CoBra/Data/dcm"))
            try:
                shutil.move(dcm_file, dst_file_dir);
            except:
                if counter==1:
                    print(f"Destination path {dst_file_dir} already exists")
        print("|",  end='')
    stop = time.time()
    print(f" {(stop-start)/60} mins")

