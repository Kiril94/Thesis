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
from utilities import stats
from utilities import basic


#%% 
# In[tables directories]
script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parent
disk_dir = "G:"
dst_data_dir = f"{disk_dir}/CoBra/Data"
dst_dcm_dir = join(dst_data_dir, 'dcm')
src_dir = join(dst_data_dir, 'GE')
batch_dirs = sorted([join(src_dir,x) for x \
                       in os.listdir(src_dir) if x.startswith('batch')])
download_pat_path = join(base_dir, "data/patient_groups")
table_dir = join(base_dir, 'data', 'tables')
patient_dir_df = pd.read_csv(join(table_dir, 'patient_directories.csv'))
patient_dir_dic = pd.Series(
    patient_dir_df.Directory.values, index=patient_dir_df.PatientID)\
        .to_dict()
#%% 
# In[move]

for batch_dir in batch_dirs:
    print(f"Batch: {batch_dir}")
    log_str = f"Batch: {batch_dir}\n"
    with open(f"{base_dir}/logs/ge_patient_log.txt", mode="a+") as f:
            f.write(log_str)
    for ge_patient_dir in basic.list_subdir(batch_dir):
        print(f"\nPatient: {ge_patient_dir}")
        dst_patient_dir = patient_dir_dic[split(ge_patient_dir)[1]]
        log_str = f"patient: {patient_dir}\n{datetime.now()}\n"
        with open(f"{base_dir}/logs/ge_patient_log.txt", mode="a+") as f:
            f.write(log_str)
        full_dst_patient_dir = join(dst_dcm_dir, dst_patient_dir)
        print(f"Copy to {full_dst_patient_dir}")
        if not os.path.exists(join(full_dst_patient_dir, 'DOC')):
            os.makedirs(join(full_dst_patient_dir, 'DOC'))
        print("Copy documentation files")
        for doc_path_src in glob.iglob(f"{ge_patient_dir}/*/DOC/*/*.pdf"):
            doc_path_src = os.path.normpath(doc_path_src)
            study_id = doc_path_src.split(os.sep)[3]
            doc_id = doc_path_src.split(os.sep)[5]
            dst_doc_dir = join(full_dst_patient_dir, 'DOC')
            doc_path_dst = join(dst_doc_dir, f"{study_id}_{doc_id}.pdf")
            try:
                shutil.copy(doc_path_src, doc_path_dst)
            except Exception as e:
                print("ERROR : "+str(e))   
        # copy dcm files
        print("Copy volumes")
        for volume_src in glob.iglob(f"{ge_patient_dir}/*/MR/*"):
            if len(os.listdir(volume_src))==0:
                print('-',  end='')
                continue
            else:        
                series_uid = volume_src.split(os.sep)[-1]
                volume_dst = join(full_dst_patient_dir, series_uid)
                try:
                    with open(f"{base_dir}/logs/ge_volume_log.txt", mode="w") as f:
                        f.write(volume_dst)
                    shutil.copytree(volume_src, volume_dst)
                    print("|",  end='')
                except Exception as e:
                    print("ERROR : "+str(e))
        print(f"Patient finished: {datetime.now()}")
    print(f"Batch {batch_dir} finished")   