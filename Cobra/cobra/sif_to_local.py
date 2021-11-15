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
disk_dir = "D:/F/"
dst_data_dir = f"{disk_dir}/CoBra/Data"
download_pat_path = join(base_dir, "data/share/Cerebriu/patient_groups")
table_dir = join(base_dir, 'data', 'tables')
#%% 
# In[Load df]

volume_dir_df = pd.read_csv(join(table_dir, 'series_directories_sif.csv'))
patient_dir_df = pd.read_csv(join(table_dir, 'patient_directories.csv'))
volume_dir_dic = pd.Series(
    volume_dir_df.Directory.values, index=volume_dir_df.SeriesInstanceUID)\
        .to_dict()
patient_dir_dic = pd.Series(
    patient_dir_df.Directory.values, index=patient_dir_df.PatientID)\
        .to_dict()
df_all = pd.read_csv(join(table_dir, "neg_pos.csv"))

#%%

print('lets download t1 pre post')
group_list = np.loadtxt(join(base_dir, "data/patient_groups","t1_pre_post.txt"),
                                   dtype='str')
df_group = df_all[df_all['PatientID'].isin(group_list)]
df_group = df_group.sort_values('PatientID')

#%%
# In[get index of last patient, remove last volume]

patient_list_group = df_group.PatientID.unique()
with open(f"{base_dir}/volume_log.txt") as f:
    series_lines = f.readlines()
last_series_path = series_lines[0]
print(f"Remove {last_series_path}")
try:
    shutil.rmtree(last_series_path)
except Exception as e:
	print("ERROR : "+str(e))

with open(f"{base_dir}/patient_log.txt") as f:
    lines = f.readlines()
last_patient_idx = int(lines[-2][6:11]) 
print(f"Patient Num: {last_patient_idx}")

#%%
# In[move local]
crb_dst = join(dst_data_dir, 'dcm')
counter = last_patient_idx
for pat in patient_list_group[last_patient_idx:]:
    patient_dir = patient_dir_dic[pat]
    counter += 1
    log_str = f"{patient_dir}\nindex: {counter}\
            \n {datetime.now()}\n"
    with open(f"{base_dir}/patient_log.txt", mode="a+") as f:
        f.write(log_str)
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

    print(f"Download {len(volumes)} volumes")
    for volume in volumes:
        try:
            volume_dir = volume_dir_dic[volume]
        except:
            print("Volume not in dict")
            continue
        volume_src = os.path.normpath(f"Y:/{volume_dir}")
        if len(os.listdir(volume_src))==0:
            print('-',  end='')
            continue
        else:        
            series_uid = volume_src.split(os.sep)[-1]
            volume_dst = join(crb_dst, patient_dir, series_uid)
            try:
                with open(f"{base_dir}/volume_log.txt", mode="w") as f:
                    f.write(volume_dst)
                shutil.copytree(volume_src, volume_dst)
                print("|",  end='')
            except Exception as e:
	            print("ERROR : "+str(e))
    stop = time.time()
    print(f"\n {(stop-start)/60:.3} min")
    