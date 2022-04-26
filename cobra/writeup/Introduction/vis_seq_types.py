#%%
import os, sys
import pickle
from os.path import split, join
import gzip
base_dir = split(split(os.getcwd())[0])[0]
if base_dir not in sys.path:
    sys.path.append(base_dir)
import pandas as pd
import numpy as np
#%%
table_dir = f"{base_dir}/data/tables"
with open(join(table_dir, 'scan_tables','scan_after_sq_pred_dst_nos_date.pkl'), 'rb') as f:
    dfc = pickle.load(f)
with open(join(table_dir, 'sif_series_directories.pkl'), 'rb') as f:
    dir_dic = pickle.load(f)

#%%
dfc.groupby('PatientID').Sequence.unique()

#%%
df1p = dfc[dfc.PatientID=='fffe96e6d67a6994994ad75d69f001d6']
t1s = df1p[df1p.Sequence=='t1'].iloc[0,0]
dir_dic[t1s]