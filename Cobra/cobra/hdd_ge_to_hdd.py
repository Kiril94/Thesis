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

for batch_dir in batch_dirs[:1]:
    print(f"Batch: {batch_dir}")
    log_str = f"Batch: {batch_dir}\n"
    with open(f"{base_dir}/ge_patient_log.txt", mode="a+") as f:
            f.write(log_str)
    for ge_patient_dir in basic.list_subdir(batch_dir)[10:12]:
        print(f"Patient: {ge_patient_dir}")
        dst_patient_dir = patient_dir_dic[split(ge_patient_dir)[1]]
        log_str = f"patient: {patient_dir}\n{datetime.now()}\n"
        with open(f"{base_dir}/ge_patient_log.txt", mode="a+") as f:
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
        for volume_src in glob.iglob(f"{ge_patient_dir}/*/MR/*"):
            if len(os.listdir(volume_src))==0:
                print('-',  end='')
                continue
            else:        
                series_uid = volume_src.split(os.sep)[-1]
                volume_dst = join(full_dst_patient_dir, series_uid)
                try:
                    with open(f"{base_dir}/ge_volume_log.txt", mode="w") as f:
                        f.write(volume_dst)
                    shutil.copytree(volume_src, volume_dst)
                    print("|",  end='')
                except Exception as e:
                    print("ERROR : "+str(e))
       

#%%
# In[move crb]
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
    print(datetime.now())
    # Copy doc files
    if not os.path.exists(join(crb_dst, patient_dir, 'DOC')):
        os.makedirs(join(crb_dst, patient_dir, 'DOC'))
    for doc_path_src in glob.iglob(f"Y:/{patient_dir}/*/DOC/*/*.pdf"):
        doc_path_src = os.path.normpath(doc_path_src)
        study_id = doc_path_src.split(os.sep)[3]
        doc_id = doc_path_src.split(os.sep)[5]
        dst_doc_dir = join(crb_dst, patient_dir, 'DOC')
        doc_path_dst = join(dst_doc_dir, f"{study_id}_{doc_id}.pdf")
        try:
            shutil.copy(doc_path_src, doc_path_dst)
        except Exception as e:
	        print("ERROR : "+str(e))
        
    # copy dcm files
    volumes = df_group[df_group.PatientID==pat]['SeriesInstanceUID']
    print(f"download {len(volumes)} volumes")
    for volume in volumes:
        try:
            volume_dir = volume_dir_dic[volume]
        except:
            print("volume not in dict")
            continue
        try:
            volume_src = os.path.normpath(f"Y:/{volume_dir}")
        except Exception as e:
            print("ERROR : "+str(e))
            continue
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
"""
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
"""    