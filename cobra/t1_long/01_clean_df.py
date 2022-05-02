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
dfc = dfc.drop_duplicates()
#%%
len(dfc)
print(dfc.PatientID.nunique())
#%%
# In[Create chart]
def print_pos_neg_scans(df):
    dfp = df[df.days_since_test>=-4]
    dfn = df[(df.days_since_test<-30) | df.days_since_test.isna()]
    print(len(df[df.days_since_test<=-4]))

def print_pos_neg(df):
    print('neg',df[df.Positive==0].PatientID.nunique())
    print('pos', df[df.Positive==1].PatientID.nunique())
print('initial length:', len(dfc))

#%%
df_d = dfc[(dfc['2019']==1)|(~dfc.InstanceCreationDate.isna())]
print(df_d.PatientID.nunique())
print(len(df_d))
#%%

df1 = df_d[df_d.Sequence=='t1']
print(df1.PatientID.nunique())
print(len(df1))
#%%
# Remove scans with missing dates for those who don't have 2019 tag
df1 = df1[(df1['2019']==1)|(~df1.InstanceCreationDate.isna())]
mask = (df1.InstanceCreationDate.isna()) & (df1['2019']==1)
#df1['InstanceCreationDate'] = np.where(
#    mask, pd.Timestamp('2018-01-01'), df1['InstanceCreationDate'])
df1.loc[mask, 'InstanceCreationDate'] = np.datetime64(datetime.datetime(2018,1,1))
#df1.InstanceCreationDate = df1.InstanceCreationDate.astype(pd.dateti)
mask3d = (df1.MRAcquisitionType.isna() | (df1.MRAcquisitionType=='3D'))\
    & (df1.NumberOfSlices>=64)
df3d1 = df1[mask3d]
print(df3d1.PatientID.nunique())
print(len(df3d1))
#%%
neg_pat0 = df3d1[df3d1.InstanceCreationDate<=datetime.datetime(2020,2,27)]
print_pos_neg(neg_pat0)

#%%
neg_pat1 = df3d1[df3d1.days_since_test.isna() | (df3d1.days_since_test<=-30)]
print_pos_neg(neg_pat1)
#%%
df3d1[~df3d1.days_since_test.isna() | (df3d1.InstanceCreationDate>=datetime.datetime(2020,2,27))]
#%%
# In[Set all missing dates with 2019 tag to 2018,1,1]
#%%
sids_3dt1 = df3d1.SeriesInstanceUID.tolist()
with open(join(base_dir, 'data', 't1_cross','3dt1_sids2.pkl'), 'wb') as f:
    pickle.dump(sids_3dt1, f)
#%%

with gzip.open(join(table_dir, 'scan_3dt1_clean.gz'), 'wb') as f:
    pickle.dump(df3d1, f)
#%%
df3d1.to_csv(join(table_dir, 'scan_3dt1_clean.csv'), index=False, header=True)