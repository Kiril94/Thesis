#%%
import pickle as pkl
import os, sys
from os.path import join, split
from pathlib import Path
import pandas as pd
script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parent.parent
data_dir = join(base_dir, 'data')
table_dir = join(data_dir, 'tables')
import json
#%%
with open(join(table_dir, "disk_series_directories.json"), "r") as json_file:
    disk_dcm_dir_dic = json.load(json_file)
with open(join(table_dir, 'scan_tables','scan_after_sq_pred_dst_nos_date.pkl'), 'rb') as f:
    dfc = pkl.load(f)
dfc = dfc.drop_duplicates()
dfc = dfc[['PatientID','SeriesInstanceUID','InstanceCreationDate','Sequence']]
with open(join(data_dir, "t1_longitudinal\increase_sids.pkl"), 'rb') as f:
    sids = pkl.load(f)
dfc = dfc[dfc.SeriesInstanceUID.isin(sids)]
#%%
dfc1 = dfc.reset_index()

one_patient = dfc1[dfc1.PatientID==dfc1.loc[0].PatientID]
print(one_patient[['InstanceCreationDate', 'SeriesInstanceUID', 'Sequence']])
#print(disk_dcm_dir_dic[dfc1.SeriesInstanceUID[8]])

exclude_pats = ['1c5310eb90ac848427226b6dcb3e401f']

#%%
print(dfc1.loc[8].PatientID)