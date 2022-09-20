"""Checking for SWI(haemorrhages) in combination with 3D T1"""
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%%
scans_path = "C:\\Users\\fjn197\\OneDrive - University of Copenhagen\\Cobra\\tables\\scan_after_sq_pred_dst_nos_date.csv"
seg_path = "C:\\Users\\fjn197\\OneDrive - University of Copenhagen\\Cobra\\tables\\volume_prediction_results_new.csv"
dfs = pd.read_csv(scans_path, usecols=['SeriesInstanceUID',
    'InstanceCreationDate', 'MRAcquisitionType', 
    'Sequence', 'NumberOfSlices'], 
    parse_dates=['InstanceCreationDate']).drop_duplicates()
print(len(dfs))
dfv = pd.read_csv(seg_path, usecols=['SeriesInstanceUID']).drop_duplicates()

#%%
print(len(dfv), 'Segmented scans')
# select 3dt1 scans
df3dt1 = dfs.query("Sequence == 't1' and \
    MRAcquisitionType != '2D' and NumberOfSlices >= 64")
print(len(df3dt1), 'classified as 3D T1')
df3dt1_d = df3dt1[~(df3dt1.InstanceCreationDate.isnull() |\
    (df3dt1.InstanceCreationDate==np.datetime64('2018-01-01')))]
print('of those ', len(df3dt1_d), 'with date')
df_ns = df3dt1[~df3dt1.SeriesInstanceUID.isin(dfv.SeriesInstanceUID)]

#%%
# plot date distribution
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
df_ns.InstanceCreationDate.hist(bins=100)
plt.show()