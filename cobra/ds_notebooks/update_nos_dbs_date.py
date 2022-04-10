#%%
import sys, os
from os.path import split, join
base_dir = split(os.getcwd())[0]
if base_dir not in sys.path:
    sys.path.append(base_dir)
from os.path import join
import pickle
import numpy as np
import datetime
#%%
fig_dir = join(base_dir, 'figs')
table_dir = join(base_dir, 'data/tables')
with open(join(table_dir, 'scan_tables','scan_after_sq_pred_dst.pkl'), 'rb') as f:
    dfc = pickle.load(f)

#%%
# df_nos = dfc[((dfc.MRAcquisitionType=='3D') | (dfc.MRAcquisitionType.isna())) \
    # & (dfc.Sequence=='t1') & (dfc.NumberOfSlices.isna())]

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
ls_3dt1 = df3dt1.SeriesInstanceUID.tolist()
len(ls_3dt1)
with open("C:\\Users\\kiril\\Thesis\\CoBra\\cobra\\data\\t1_longitudinal\\sids_3dt1_temp.pkl", 'wb') as f:
    pickle.dump(ls_3dt1, f)
#%%
mask = (dfc.InstanceCreationDate.isna()) & (dfc['2019']==1)
dfc.loc[mask,'InstanceCreationDate'] = datetime.datetime(2018, 1, 1)
#%%
with open(join(table_dir, 'scan_tables','scan_after_sq_pred_dst_nos_date.pkl'), 'wb') as f:
    pickle.dump(dfc,f)
#%%
dfc.to_csv(join(table_dir, 'scan_tables','scan_after_sq_pred_dst_nos_date.csv'), index=False)