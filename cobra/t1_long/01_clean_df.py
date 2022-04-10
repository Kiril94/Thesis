#%%
import os, sys
import pickle
from os.path import split, join
import gzip
base_dir = split(os.getcwd())[0]
if base_dir not in sys.path:
    sys.path.append(base_dir)
from pathlib import Path
import datetime
import pandas as pd
import numpy as np
#%%
# In[dirs and load df]
script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parent.parent
table_dir = f"{base_dir}/data/tables"
with open(join(table_dir, 'scan_tables','scan_after_sq_pred_dst_nos_date.pkl'), 'rb') as f:
    dfc = pickle.load(f)
df1 = dfc[dfc.Sequence=='t1']

def print_pos_neg(df):
    print('neg',df[df.Positive==0].PatientID.nunique())
    print('pos', df[df.Positive==1].PatientID.nunique())
print('initial length:', len(df1))
print_pos_neg(df1)

#%%
# Remove scans with missing dates for those who don't have 2019 tag
df1 = df1[(df1['2019']==1)|(~df1.InstanceCreationDate.isna())]
mask = (df1.InstanceCreationDate.isna()) & (df1['2019']==1)
df1['InstanceCreationDate'] = np.where(
    mask, datetime.datetime(2018, 1, 1), df1['InstanceCreationDate'])
mask3d = (df1.MRAcquisitionType.isna() | (df1.MRAcquisitionType=='3D'))\
    & (df1.NumberOfSlices>=64)
df3d1 = df1[mask3d]
print_pos_neg(df3d1)
#%%
# In[Set all missing dates with 2019 tag to 2018,1,1]

#%%

with gzip.open(join(table_dir, 'scan_3dt1_clean.gz'), 'wb') as f:
    pickle.dump(df3d1, f)
#%%
df3d1.to_csv(join(table_dir, 'scan_3dt1_clean.csv'), index=False, header=True)