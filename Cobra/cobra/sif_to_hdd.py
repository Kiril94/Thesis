# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 10:58:38 2021

@author: klein
"""
#%% 
# Import
import shutil
import os
from os.path import join, split, exists
from pathlib import Path
import glob
import time
import numpy as np
import pandas as pd
import sys
cobra_dir = "D:/Thesis/Cobra/cobra"
if cobra_dir not in sys.path:
    sys.path.append(cobra_dir)
from utilities.utils import target_path
from utilities import stats
print("We will download only dwi, swi, flair, t1, t2, t2*")
print("Start with smallest group of patients (1104) dwi, flair, t2*, t1, mostly negative patients,")

#%%# 
# tables directories
script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parent
src_dirs = os.listdir("Y:")
src_neg_dirs = sorted([f"{src_dirs}/{x}" for x \
                       in src_dirs if x.startswith('2019')])
dst_data_dir = "G:/CoBra/Data"
download_pat_path = join(base_dir, "data/share/Cerebriu/download_patients")
# In[Load df]
volume_dir_df = pd.read_csv(join(base_dir, 'data/tables', 'sid_directories.csv'))
volume_dir_dic = pd.Series(
    volume_dir_df.Directory.values, index=volume_dir_df.SeriesInstanceUID).to_dict()
df_all = pd.read_csv(join(base_dir, "data/tables/neg_pos.csv"))
# In[Get relevant patients and volumes]
print(os.listdir(download_pat_path))
#rel_seq = ['dwi', 'swi', 't1', 't2', 't2s', 'flair']
# download all sequences
dftt_list = np.loadtxt(join(download_pat_path, "dwi_flair_t2s_t1.txt"),
                                   dtype='str')
dftt = df_all[df_all['PatientID'].isin(dftt_list)]
#df_patients_0 = df_patients_0[df_patients_0['Sequence'].isin(rel_seq)]
dftt = dftt.sort_values('PatientID')

volume_dir_df = pd.read_csv(join(base_dir, 'data/tables', 'sid_directories.csv'))
volume_dir_dic = pd.Series(
    volume_dir_df.Directory.values, index=volume_dir_df.SeriesInstanceUID).to_dict()

#%%
#  In[get index of last patient]
# 1st patient was already written
# last patient: 0385ef30676c4602159171edac0cc2d6
patient_list_dfft = dftt.PatientID.unique()
last_patient = "15473b3462554d4f81eb36caefca4978"
last_patient_idx = np.where(patient_list_dfft==last_patient)[0][0]

#%%
#  In[test]
print(patient_list_dfft[last_patient_idx:])
#%%
#  In[move crb]
for pat in patient_list_dfft[last_patient_idx:]:
    start = time.time()
    print(f"patient: {pat}:", end=' ')
    volumes = dftt[dftt.PatientID==pat]['SeriesInstanceUID']
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

#%%
#  In[move GE (Akshay)]
dwi_t2s_gre = np.loadtxt(join(download_pat_path, "ge_dwi_t2s_gre.txt"),
                                   dtype='str')
gdtg = df_all[df_all['PatientID'].isin(dwi_t2s_gre)]
gdtg = gdtg.sort_values('PatientID')
patient_list_gdtg = gdtg.PatientID.unique()
batches = []
for i in range(6):
    if i<5:
        batches.append(patient_list_gdtg[i*50:(i+1)*50])
    else:
        batches.append(patient_list_gdtg[5*50:])
print(len(batches[-1]))

    
#%%
#  In[copy whole tree]
current_batch = 1
ge_dir = os.path.normpath("G:\CoBra\Data\GE")
for i, batch in enumerate(batches[current_batch:]):
    batch_dir = join(ge_dir, f"batch_{i+current_batch}")
    for pat in batch:
        start = time.time()
        patient_dir = volume_dir_df[
            stats.check_tags(
                volume_dir_df, tags=[pat], key='Directory')].iloc[0,1]
        patient_dir = os.path.normpath(patient_dir)
        patient_dir = join(*list(patient_dir.split(os.sep)[:2]))
        patient_dir = f"Y:\{patient_dir}"
        _, patient_id = split(patient_dir)
        dst_dir = join(batch_dir, patient_id)
        if exists(dst_dir):
            print(f"{dst_dir} exists" )
            continue
        print(f"{patient_dir}\n->{dst_dir}")
        shutil.copytree(patient_dir, dst_dir)
        stop = time.time()
        print(f" {(stop-start)/60} mins")
#%%
# In[Save metadata]
ge_meta = df_all[df_all.PatientID.isin(patient_list_gdtg)]
ge_meta['batch'] = None
for i, batch in enumerate(batches):
    ge_meta.loc[ge_meta.PatientID.isin(batch), 'batch'] = i
ge_meta.to_csv("G:\CoBra\Data\GE\metadata.csv",
               index=False,encoding='utf-8-sig')
    