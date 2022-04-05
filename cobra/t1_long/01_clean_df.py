#%%
import os, sys
import pickle
from os.path import split, join
import gzip
base_dir = split(os.getcwd())[0]
if base_dir not in sys.path:
    sys.path.append(base_dir)
from pathlib import Path
from utilities import utils
import datetime
import pandas as pd
import numpy as np
#%%
# In[dirs and load df]
script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parent.parent
table_dir = f"{base_dir}/data/tables/scan_tables"
# dfc = pd.read_feather(join(base_dir,'data','tables', 'neg_pos_clean'))
with open(join(table_dir, 'scan_after_sq_pred.pkl'), 'rb') as f:
    df = pickle.load(f)
df1 = df[df.Sequence=='t1']
def print_pos_neg(df):
    print('neg',df[df.Positive==0].PatientID.nunique())
    print('pos', df[df.Positive==1].PatientID.nunique())

print('initial length:', len(df1))
print_pos_neg(df1)

#%%
# Remove scans with missing dates for those who don't have 2019 tag
df1_0 = df1[~((df1['2019']==0) & (df1.InstanceCreationDate.isna()))]
df1_0.set_index( "SeriesInstanceUID", inplace=True )
print('removing missing dates from 2020/2021:', len(df1_0))
print_pos_neg(df1_0)
#%%
df1_0.InstanceCreationDate.hist()
#%%
# In[Set all missing dates with 2019 tag to 2018,1,1]
mask = (df1_0.InstanceCreationDate.isna()) & (df1_0['2019']==1)
df1_0['InstanceCreationDate'] = np.where(
    mask, datetime.datetime(2018, 1, 1), df1_0['InstanceCreationDate'])


#%%
# In[Get number of slices if missing]
print("We need the number of slices for scans that have as MRAcquisitionType either nan or 3D")
import importlib
importlib.reload(utils)
df1_3dnone = df1_0[(df1_0.MRAcquisitionType=='3D') | (df1_0.MRAcquisitionType.isna())]
df1_3dnone_nos_miss = df1_3dnone[df1_3dnone.NumberOfSlices.isna()]
print(len(df1_3dnone_nos_miss))
# n_slices_dic = utils.save_number_of_slices_to_txt(df1_3dnone_nos_miss, 'nos.txt', 
                            # sif_path='Y://', disk_path='F:')
#%%
# In[Save dict]
# with open('nos.pkl', 'wb') as f:
    # pickle.dump(n_slices_dic, f)
# In[Load dict]
with open('nos.pkl', 'rb') as f:
    n_slices_dic = pickle.load(f)

df1_0.NumberOfSlices.fillna(pd.Series(n_slices_dic), inplace=True)

#%%
# In[Keep scan if 3D or if missing if number of slices>=64]
#df1_1 = df1_0[df1_0.MRAcquisitionType.isna() ].NumberOfSlices.hist()
mask3d = (df1_0.MRAcquisitionType.isna() | (df1_0.MRAcquisitionType=='3D'))\
    & (df1_0.NumberOfSlices>=64)
df3d = df1_0[mask3d]
df3d = df3d.reset_index()

#%%

with gzip.open(join(table_dir, 'scan_3dt1_clean.gz'), 'wb') as f:
    pickle.dump(df3d, f)
#%%
df3d.to_csv(join(table_dir, 'scan_3dt1_clean.csv'), index=False, header=True)
#%%
df3d.keys()
#%%
#df[df1_1.NumberOfSlices<200].NumberOfSlices.hist(bins=20)
print(len(df3d),'3d scans')
df3d.DistanceBetweenSlices.isna().mean()
# dfc_3dt1 = dfc[(dfc.NumberOfSlices>=64) & (dfc.MRAcquisitionType=='3D')]
# df_new_pat = df3d[~df3d.PatientID.isin(dfc_3dt1.PatientID)]

#%%
print('3D T1 patients')
print_pos_neg(df3d)
df3d.days_since_test.hist()
#%%
# In[Compute DistanceBetweenSlices for positive to start with,
#   after initial matching we can get the negatives]
