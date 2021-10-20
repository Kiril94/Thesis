# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 10:58:38 2021

@author: klein
"""
#%% 
# In[Import]
import shutil
import os
from os.path import join, split, exists
from pathlib import Path
import glob
import time
from datetime import datetime
import numpy as np
import pandas as pd
import sys
cobra_dir = "D:/Thesis/Cobra/cobra"
if cobra_dir not in sys.path:
    sys.path.append(cobra_dir)
from utilities import stats


#%% 
# In[tables directories]
script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parent
src_dirs = os.listdir("Y:/")
src_neg_dirs = sorted([f"{src_dirs}/{x}" for x \
                       in src_dirs if x.startswith('2019')])
disk_dir = "G:"
dst_data_dir = f"{disk_dir}/CoBra/Data"
download_pat_path = join(base_dir, "data/share/Cerebriu/patient_groups")
table_dir = join(base_dir, 'data', 'tables')
#%% 
# In[Load df]

volume_dir_df = pd.read_csv(join(table_dir, 'sid_directories.csv'))
patient_dir_df = pd.read_csv(join(table_dir, 'patient_directories.csv'))
volume_dir_dic = pd.Series(
    volume_dir_df.Directory.values, index=volume_dir_df.SeriesInstanceUID)\
        .to_dict()
patient_dir_dic = pd.Series(
    patient_dir_df.Directory.values, index=patient_dir_df.PatientID)\
        .to_dict()
df_all = pd.read_csv(join(table_dir, "neg_pos.csv"))

#%%
# In[Get relevant patients and volumes]
print("Start with smallest group of patients (1104) dwi, \
    flair, t2*, t1, mostly negative patients,")
dftt_list = np.loadtxt(join(download_pat_path, "dwi_flair_t2s_t1.txt"),
                                   dtype='str')
df_group = df_all[df_all['PatientID'].isin(dftt_list)]
# In case you want to download only specific sequences uncomment next lines
# rel_seq = ['dwi', 'swi', 't1', 't2', 't2s', 'flair']
# df_group = df_group[df_group['Sequence'].isin(rel_seq)]
df_group = df_group.sort_values('PatientID')

#%%
# In[get index of last patient]
patient_list_group = df_group.PatientID.unique()
# if you want to start with a specific patient uncomment and set last_patient
#last_patient = "15473b3462554d4f81eb36caefca4978"
#last_patient_idx = np.where(patient_list_group==last_patient)[0][0]

#%%
# In[move crb]
crb_dst = join(dst_data_dir, 'dcm')
for pat in patient_list_group[:]:
    patient_dir = patient_dir_dic[pat]
    start = time.time()
    print(f"Patient: {patient_dir}", end='\n')
    print(datetime.now().strftime("%H:%M:%S"))
    # Copy doc files
    if not os.path.exists(join(crb_dst, patient_dir, 'DOC')):
        os.makedirs(join(crb_dst, patient_dir, 'DOC'))
    for doc_path_src in glob.iglob(f"Y:/{patient_dir}/*/DOC/*/*.pdf"):
        doc_path_src = os.path.normpath(doc_path_src)
        study_id = doc_path_src.split(os.sep)[3]
        doc_id = doc_path_src.split(os.sep)[5]
        dst_doc_dir = join(crb_dst, patient_dir, 'DOC')
        doc_path_dst = join(dst_doc_dir, f"{study_id}_{doc_id}.pdf")
        shutil.copy(doc_path_src, doc_path_dst)
    # copy dcm files
    volumes = df_group[df_group.PatientID==pat]['SeriesInstanceUID']
    print(f"download {len(volumes)} volumes")
    for volume in volumes:
        volume_dir = volume_dir_dic[volume]
        counter = 0
        volume_src = os.path.normpath(f"Y:/{volume_dir}")
        if len(os.listdir(volume_src))==0:
            print('-')
            continue
        else:        
            series_uid = volume_src.split(os.sep)[-1]
            volume_dst = join(crb_dst, patient_dir, series_uid)
            shutil.copytree(volume_src, volume_dst)
            print("|",  end='')
    stop = time.time()
    print(f" {(stop-start)/60:.2} min")

#%%
# In[move GE (Akshay)]
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
# In[copy whole tree]
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
    