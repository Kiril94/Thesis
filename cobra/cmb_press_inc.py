"""Checking for SWI(haemorrhages) in combination with 3D T1"""
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%%
scans_path = "C:\\Users\\fjn197\\OneDrive - University of Copenhagen\\Cobra\\tables\\scan_after_sq_pred_dst_nos_date.csv"
seg_path = "C:\\Users\\fjn197\\OneDrive - University of Copenhagen\\Cobra\\tables\\volume_prediction_results_new.csv"
dfs = pd.read_csv(scans_path, usecols=['PatientID',
    'SeriesInstanceUID',
    'InstanceCreationDate', 'MRAcquisitionType', 
    'Sequence', 'NumberOfSlices'], 
    parse_dates=['InstanceCreationDate']).drop_duplicates()
print(len(dfs))
dfv = pd.read_csv(seg_path, usecols=['SeriesInstanceUID']).drop_duplicates()
dfv = pd.merge(dfv, dfs[['PatientID','SeriesInstanceUID']], 
    how='inner', on='SeriesInstanceUID')
#%%
print(len(dfv), 'Segmented scans')
# select 3dt1 scans
df3dt1 = dfs.query("Sequence == 't1' and \
    MRAcquisitionType != '2D' and NumberOfSlices >= 64")
print(len(df3dt1), 'classified as 3D T1')
df3dt1_d = df3dt1[~(df3dt1.InstanceCreationDate.isnull() |\
    (df3dt1.InstanceCreationDate==np.datetime64('2018-01-01')))]
print('of those', len(df3dt1_d), 'have a date')
df_ns = df3dt1_d[~df3dt1_d.SeriesInstanceUID.isin(dfv.SeriesInstanceUID)]
print(len(df_ns), '3D T1 scans without segmentation')
df_m = pd.merge(df3dt1_d, dfv, how='outer',indicator=True)


#%%
print('right: only segmented, not classified as 3D T1')
print('left: only classified as 3D T1, not segmented')
df_m['_merge'] = df_m._merge.astype('category')
print(df_m._merge.value_counts())

#%%
# plot date distribution
sns.set()
# limit time range
ax = df3dt1_d.InstanceCreationDate.hist(bins=20, 
    range=(np.datetime64('2018-12-01'), np.datetime64('2020-07-01')))
ax.tick_params(axis='x', rotation=45)
#plot date distribution within time range
plt.show()
#%%
# Now look at SWI scans
dfswi = dfs.query("Sequence == 'swi'")
print(len(dfswi), 'SWI scans')
dfswi = dfswi[~dfswi.InstanceCreationDate.isnull()]
print(len(dfswi), 'SWI scans with date')
print('This is probably a clean df where all scans have a date')
# check date difference of 3D T1 and SWI scans after patient groupby
#%%
# check for combinations of swi and 3dt1
df_swi_3dt1 = pd.merge(dfswi, dfv, how='inner', on='PatientID')
print(df_swi_3dt1.PatientID.nunique(), 'SWI and 3D T1 scans')
print('Get first SWI and last 3D T1 scan and check \
    for which Patients SWI was performed before the 3D T1 scan')
print(dfswi.PatientID.nunique(), 'Patients with SWI scans')
dfswi_min = dfswi.loc[dfswi.groupby('PatientID').InstanceCreationDate.idxmin()]
df3dt1_max = df3dt1_d.loc[df3dt1_d.groupby('PatientID').InstanceCreationDate.idxmax()]
df_swi_3dt1 = pd.merge(dfswi_min, df3dt1_max, how='inner', on='PatientID',
    suffixes=('_swi', '_3dt1'))

#df['VALUE'] = df['VALUE'].where(df.groupby(['ID_1', 'ID_2'])['DATE'].transform('min').eq(df['DATE']))