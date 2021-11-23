#!python
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
from utilities import download

#%% 
# In[tables directories]
script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parent
src_dirs = os.listdir("Y:/")
src_neg_dirs = sorted([f"{src_dirs}/{x}" for x \
                       in src_dirs if x.startswith('2019')])
disk_dir = "D:/F/"
dst_data_dir = f"{disk_dir}/CoBra/Data/dcm"
download_pat_path = join(base_dir, "data/share/Cerebriu/patient_groups")
table_dir = join(base_dir, 'data', 'tables')
#%% 
# In[Load df]
print("Load dataframes")
df_volume_dir = pd.read_csv(join(table_dir, 'series_directories_sif.csv'))
df_patient_dir = pd.read_csv(join(table_dir, 'patient_directories.csv'))
df_all = pd.read_csv(join(table_dir, "neg_pos.csv"))
print("Load dataframes finished")
#%%
print('lets download t1 neg')
group_list = np.loadtxt(join(base_dir, "data/patient_groups","t1_post.txt"),
                                   dtype='str')
df_group = df_all[df_all['PatientID'].isin(group_list)]
rel_seq = ['t1']
if len(rel_seq)>0:
    print(f"Only the sequences {rel_seq} will be downloaded.")
    df_group = df_group[df_group['Sequence'].isin(rel_seq)]
df_group = df_group.sort_values('PatientID')

#%%
# In[get index of last patient, remove last volume]
download.move_files_from_sif(df_group, df_volume_dir, df_patient_dir, 
                        dst_data_dir, patient_log_file, volume_log_file)
