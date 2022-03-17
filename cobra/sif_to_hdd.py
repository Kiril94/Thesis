# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 10:58:38 2021
@author: klein
"""
#%% 
# In[Import]
import os
from os.path import join
from pathlib import Path
import pandas as pd
from utilities import download
import pickle

#%% 
# In[tables directories]
script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parent
disk_dir = "F:"
dst_data_dir = f"{disk_dir}/CoBra/Data/dcm"
data_dir = join(base_dir, 'data')
table_dir = join(data_dir, 'tables')
update_downloaded_files = False


#%% 
# In[Load df]
print("Load dataframes")
df_volume_dir = pd.read_csv(join(table_dir, 'series_directories.csv'))
df_patient_dir = pd.read_csv(join(table_dir, 'patient_directories.csv'))
df_all = pd.read_csv(join(table_dir, "neg_pos_clean.csv"))
print("Load dataframes finished")


sids_file_name = "3dt1_sids"
sids_3d_t1_path = join(data_dir, 't1_longitudinal', f'{sids_file_name}.pkl')
print("Using sids from the file: ", sids_3d_t1_path )
with open(sids_3d_t1_path, 'rb') as f:
    sids_3d_t1_ls = pickle.load(f)
#%%
# In[Get relevant patients and volumes]
# This batch is not finished yet, however we will first take 
# the one without t2s, since positive patients have no t2s
#print("Start with smallest group of patients (1104) dwi, \
#    flair, t2*, t1, mostly negative patients,")
#group_list = np.loadtxt(join(download_pat_path, "dwi_flair_t2s_t1.txt"),
#                                   dtype='str')


#pick only last 280 files
#df_group = df_group.iloc[-260:]
#print("Download the group t1")sids
#group_list = np.loadtxt(join(download_pat_path, "t1_neg_0.txt"),
#                                   dtype='str')
group_list = sids_3d_t1_ls
if update_downloaded_files:
    print("Save list of already downloaded volumes")
    num_pat, num_vol = download.save_list_downloaded_volumes_and_patients()
    print(num_pat, "Patients and", num_vol, "volumes already downloaded")
downloaded_ls = download.get_downloaded_volumes_ls()
print('interesection', len(set(group_list).intersection(set(downloaded_ls))))
group_list = list(set(group_list).difference(set(downloaded_ls)))

#print('Download 3dt1 scans that occur in pairs')
print("Volumes still to download: ", len(group_list))
use_batches = True
if use_batches:
    batch = 6
    print("batch:", batch)
    start = 0
    batch_size = 1000
    df_group = df_all[df_all.SeriesInstanceUID.isin(group_list[start+batch*batch_size:start+batch_size*(batch+1)])]
else:
    df_group = df_all[df_all.SeriesInstanceUID.isin(group_list)]
df_group = df_group.sort_values('PatientID')                            

print("Move ", len(df_group), "Volumes")
print("Move ", df_group.PatientID.nunique(), "Patients")

#%%
# In[Move]
if use_batches:
    patient_log_file = join(base_dir, 'logs', f"{sids_file_name}_patient_log_{batch}.txt" )
    volume_log_file = join(base_dir, 'logs', f"{sids_file_name}_volume_log_{batch}.txt" )
else:
    patient_log_file = join(base_dir, 'logs', f"{sids_file_name}_patient_log.txt" )
    volume_log_file = join(base_dir, 'logs', f"{sids_file_name}_volume_log.txt" )
download.move_files_from_sif(df_group, df_volume_dir, df_patient_dir, 
                        dst_data_dir, patient_log_file, volume_log_file, src_dir="Y:\\")

#%%
# In[move GE ]
"""dwi_t2s_gre = np.loadtxt(join(download_pat_path, "ge_dwi_t2s_gre.txt"),
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

"""
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


