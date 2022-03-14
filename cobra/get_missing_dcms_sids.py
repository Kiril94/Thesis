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
import pickle
import json
from time import time
#%% 
# In[tables directories]
script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parent
disk_dir = "F:"
sif_dir = "Y:\\"
dst_data_dir = f"{disk_dir}/CoBra/Data/dcm"
data_dir = join(base_dir, 'data')
table_dir = join(data_dir, 'tables')
update_downloaded_files = False
pred_dir = join(disk_dir, "CoBra\\Data\\volume_longitudinal_nii\\prediction")
data_long_dir = join(data_dir, 't1_longitudinal')
results_dir = join(data_long_dir, 'results')
#%% 
with open(join(table_dir, 'sif_series_directories.pkl'), 'rb') as f:
    sif_volume_dir_dic = pickle.load(f)
with open(join(data_long_dir, "sids_long_new.pkl"), 'rb') as f:
    long_sids_ls = pickle.load(f)
with open(join(table_dir, "disk_series_directories.json"), 'rb') as f:
    dir_dic = json.load(f)

#%%
def check_dcm(sid):
    disk_sid_dir = dir_dic[sid]
    sif_sid_dir = join(sif_dir, sif_volume_dir_dic[sid])
    ldisk = len(os.listdir(disk_sid_dir))
    lsif = len(os.listdir(sif_sid_dir))
    if not ldisk==lsif:
        return sid
    else: return None

def get_missing_dcm_sids(sids):
    missing_dcms_sids = []
    for i, sid in enumerate(sids):
        if i%10==0:
            print('.')
        if check_dcm(sid):
            missing_dcms_sids.append(sid)
    return missing_dcms_sids


s = time()
missing_dcms_sids = get_missing_dcm_sids(long_sids_ls)
print("#missing dcms:", len(missing_dcms_sids))
print(time()-s)
if len(missing_dcms_sids)>0:
    with open(join(results_dir, 'missing_dcms_sids.pkl'), 'wb') as f:
        pickle.dump(missing_dcms_sids, f)
