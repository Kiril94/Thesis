# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 10:58:38 2021

@author: klein
"""
# In[import]
import shutil
import os
from os.path import join, split
from pathlib import Path
import glob
import time
import numpy as np
import pandas as pd
from utilities.utils import target_path
from utilities import stats

print("We will download only dwi, swi, flair, t1, t2, t2*")
print("Start with smallest group of patients (1104) dwi, flair, t2*, t1, mostly negative patients,")

    
# In[tables directories]
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

# In[get index of last patient]
# 1st patient was already written
# last patient: 0385ef30676c4602159171edac0cc2d6
patient_list_dfft = dftt.PatientID.unique()
last_patient = "15473b3462554d4f81eb36caefca4978"
last_patient_idx = np.where(patient_list_dfft==last_patient)[0][0]

# In[test]
print(patient_list_dfft[last_patient_idx:])
# In[move crb]
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

# In[move GE (Akshay)]
dwi_t2s_gre = np.loadtxt(join(download_pat_path, "ge_dwi_t2s_gre.txt"),
                                   dtype='str')
gdtg = df_all[df_all['PatientID'].isin(dwi_t2s_gre)]
#df_patients_0 = df_patients_0[df_patients_0['Sequence'].isin(rel_seq)]
gdtg = gdtg.sort_values('PatientID')

patient_list_gdtg = gdtg.PatientID.unique()
#last_patient = "15473b3462554d4f81eb36caefca4978"
#last_patient_idx = np.where(patient_list==last_patient)[0][0]
#print(patient_list[last_patient_idx:])
# In[]
print(volume_dir_df.SeriesInstanceUID)

# In[]

for pat in patient_list_gdtg[:1]:
    # Get patient directory
    patient_dir = volume_dir_df[
        stats.check_tags(volume_dir_df, tags=[pat], key='Directory')].iloc[0,1]
    #print(join(*list(patient_dir.iloc[0,1].split(os.sep)[:2])))
    patient_dir = os.path.normpath(patient_dir)
    patient_dir = join(*list(patient_dir.split(os.sep)[:2]))
    for doc in glob.iglob(f"Y:/{patient_dir}/*/DOC/*.pdf"):
        counter = 0
        doc_dst_file_dir = join("G:/CoBra/Data/test", patient_dir, 'DOC')
        try:
            shutil.move(doc, doc_dst_file_dir);
        except:
            if counter==1:
                print(f"Destination path {doc_dst_file_dir} already exists")
        counter += 1
    break
    start = time.time()
    print(f"patient: {pat}:", end=' ')
    volumes = gdtg[gdtg.PatientID==pat]['SeriesInstanceUID']
    print(f"{len(volumes)} volumes")
    for volume in volumes:
        volume_dir = volume_dir_dic[volume]
        counter = 0
        for dcm_file in glob.iglob(f"Y:/{volume_dir}/*"):
            counter+=1
            dst_file_dir = target_path(
                Path(os.path.split(dcm_file)[0]), Path("G:/CoBra/Data/test"))
            try:
                shutil.move(dcm_file, dst_file_dir);
            except:
                if counter==1:
                    print(f"Destination path {dst_file_dir} already exists")
            if counter==2:
                break
        print("|",  end='')
    stop = time.time()
    print(f" {(stop-start)/60} mins")