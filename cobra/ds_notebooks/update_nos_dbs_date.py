#%%
import sys, os
from os.path import split, join
base_dir = split(os.getcwd())[0]
if base_dir not in sys.path:
    sys.path.append(base_dir)
from os.path import join
import pickle
import numpy as np
import pandas as pd
import datetime
#%%
fig_dir = join(base_dir, 'figs')
table_dir = join(base_dir, 'data/tables')
with open(join(table_dir, 'scan_tables','scan_after_sq_pred_dst.pkl'), 'rb') as f:
    dfc = pickle.load(f)

#%%
# In[Add distance between slices]

df_dbs = pd.read_csv('C:\\Users\\kiril\\Thesis\\CoBra\\cobra\\data\\t1_cross\\distance_between_slices\\all_distances.txt', 
    header=None, delimiter=' ', names=['SeriesInstanceUID','DistanceBetweenSlices'])
dfc = pd.merge(dfc.drop(columns=['DistanceBetweenSlices']), df_dbs, on='SeriesInstanceUID', how='left')
print(dfc.keys())
# df_nos = dfc[((dfc.MRAcquisitionType=='3D') | (dfc.MRAcquisitionType.isna())) \
    # & (dfc.Sequence=='t1') & (dfc.NumberOfSlices.isna())]
#print(dfc.DistanceBetweenSlices)
# nos_dic2 = utils.save_number_of_slices_to_txt(df_nos, 'nos2.txt', 'Y:\\', 'F:\\')
# with open("nos2.pkl", 'wb') as f:
    # pickle.dump(nos_dic2, f)
#%% 
# In[Add number of slices for 3dt1]
with open('nos3dt1.pkl', 'rb') as f:
    nos = pickle.load(f)
with open('nos23dt1.pkl', 'rb') as f:
    nos2 = pickle.load(f)
dfc.NumberOfSlices = dfc.NumberOfSlices.fillna(dfc.SeriesInstanceUID.map(nos))
dfc.NumberOfSlices = dfc.NumberOfSlices.fillna(dfc.SeriesInstanceUID.map(nos2))
#%%
# In[Save 3dt1 for download]
df3dt1 = dfc[((dfc.MRAcquisitionType=='3D') | (dfc.MRAcquisitionType.isna())) \
    & ((dfc.Sequence=='t1') & (dfc.NumberOfSlices>=64))]
df3dt1 = df3dt1[(df3dt1['2019']==1)|(~df3dt1.InstanceCreationDate.isna())]
df3dt1 = df3dt1[df3dt1.days_since_test.isna() |\
      (df3dt1.days_since_test>=-3) | (df3dt1.days_since_test<=-30)]
ls_3dt1 = df3dt1.SeriesInstanceUID.tolist()
print(len(ls_3dt1))
with open("C:\\Users\\kiril\\Thesis\\CoBra\\cobra\\data\\t1_longitudinal\\sids_3dt1_temp.pkl", 'wb') as f:
    pickle.dump(ls_3dt1, f)
#%%
mask = (dfc.InstanceCreationDate.isna()) & (dfc['2019']==1)
dfc.loc[mask,'InstanceCreationDate'] = datetime.datetime(2018, 1, 1)
#%%
with open(join(table_dir, 'scan_tables','scan_after_sq_pred_dst_nos_date.pkl'), 'wb') as f:
    pickle.dump(dfc,f)
#%%
dfc = dfc.drop_duplicates()
dfc.to_csv(join(table_dir, 'scan_tables','scan_after_sq_pred_dst_nos_date.csv'), index=False)
#%%
