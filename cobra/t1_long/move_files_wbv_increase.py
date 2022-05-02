#%%
# In[Import]
import os, sys
from os.path import join, split
from pathlib import Path
import pandas as pd
base_dir = os.getcwd()
print(base_dir)
if base_dir not in sys.path:
    sys.path.append(base_dir)
from utilities import download
import pickle
import json
from pathlib import Path
import shutil
from datetime  import datetime as dt
#%%
script_dir = os.path.realpath(__file__)
table_dir = join(base_dir, 'data','tables')
disk_dir = "F:"
with open(join(table_dir, "disk_series_directories.json"), "r") as json_file:
    disk_dcm_dir_dic = json.load(json_file)
src_data_dir = f"{disk_dir}/CoBra/Data/dcm"
tgt_base_dir = "F:\\CoBra\\Data\\volume_longitudinal_nii\\brain_volume_increase_examples"
#%%

with open("C:\\Users\\kiril\\Thesis\\CoBra\\cobra\\data\\t1_longitudinal\\increase_sids.pkl", 'rb') as f:
    group_list = pickle.load(f)
df_pat = pd.read_csv("F:\\CoBra\\Data\\volume_longitudinal_nii\\brain_volume_increase_examples\\wb_increase.csv")
with open(join(table_dir, 'scan_tables','scan_after_sq_pred_dst_nos_date.pkl'), 'rb') as f:
    dfc = pickle.load(f)
dfc = dfc.drop_duplicates()
dfc = dfc[['PatientID', 'SeriesInstanceUID','InstanceCreationDate']]
dfc = dfc[dfc.SeriesInstanceUID.isin(group_list)]
#%%
for pid in df_pat.PatientID.unique():
    print('-', end='')
    dfcp = dfc[dfc.PatientID==pid]
    dates = df_pat[df_pat.PatientID==pid].InstanceCreationDate
    date1 = dt.strptime(dates.iloc[0], '%Y-%m-%d')
    date2 = dt.strptime(dates.iloc[1], '%Y-%m-%d')
    i=0
    for index, row in dfcp.iterrows(): 
        sid = row.SeriesInstanceUID
        try:
            src_dir = disk_dcm_dir_dic[sid]
        except Exception as e: 
            print(e)
            continue
        if i==0:
            doc_dir = join(split(src_dir)[0], 'DOC')
            if os.path.exists(doc_dir):
                if not os.path.exists(join(tgt_base_dir, pid, 'DOC')):
                    shutil.copytree(doc_dir, join(tgt_base_dir, pid, 'DOC'))
                print('d', end='')
            i+=1
        if row.InstanceCreationDate==date1:
            tgt_dir = join(tgt_base_dir, pid,'study1',sid)
            if os.path.exists(tgt_dir):
                continue
            shutil.copytree(src_dir, tgt_dir)    
            print('.', end='')
        elif row.InstanceCreationDate==date2:
            src_dir = disk_dcm_dir_dic[sid]
            tgt_dir = join(tgt_base_dir, pid,'study2',sid)
            if os.path.exists(tgt_dir):
                continue
            shutil.copytree(src_dir, tgt_dir)
            print('.', end='')
        else:
            print('Dates do not correspond')
    #date2 = dt(dates.iloc[1])
    #print(pat)
    #print(disk_dcm_dir_dic[group_list[0]])