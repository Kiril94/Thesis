#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_pickle("C:\\Users\\kiril\\Thesis\\CoBra\\cobra\\data\\tables\\scan_tables\\scan_after_sq_pred_dst_nos_date.pkl")

#%%
df = df[['PatientID','InstanceCreationDate','Sequence', 'MRAcquisitionType', 'NumberOfSlices', 'Positive', 'days_since_test']]
df = df[~df.InstanceCreationDate.isna()]
df19 = df[df.InstanceCreationDate < '2020-01-01']

# groupby Sequence and plot bar unique patient ids
fig, ax = plt.subplots(2,2, figsize=(12,7))
# different order of ticks
ax = ax.flatten()
df19.groupby('Sequence').PatientID.nunique().plot(kind='bar', ax=ax[1])
df19.Sequence.value_counts().plot(kind='bar', ax=ax[0])  
ax[0].set_xlabel('Sequence')
ax[0].set_ylabel('Number of scans')
ax[1].set_ylabel('Number of patients')

df3d19 = df19[df19.MRAcquisitionType == '3D']
df3d19.groupby('Sequence').PatientID.nunique().plot(kind='bar', ax=ax[3])
df3d19.Sequence.value_counts().plot(kind='bar', ax=ax[2])  
ax[2].set_xlabel('Sequence')
ax[2].set_ylabel('Number of 3D scans')
ax[3].set_ylabel('Number of patients with 3D scans')

fig.suptitle('2019')
fig.savefig('C:\\Users\\kiril\\Thesis\\CoBra\\cobra\\figs\\scan_and_patient_nums_2019.png', dpi=300)

#%%
data = np.array([])


df.InstanceCreationDate.isna().sum()
#%%
df_nums = pd.DataFrame()
df_nums.insert(0, '#Patients', df19.groupby('Sequence').PatientID.nunique())
df_nums.insert(1, '#Scans', df19.Sequence.value_counts())
df_nums.insert(2, '3dT1 #Patients', df3d19.groupby('Sequence').PatientID.nunique())
df_nums.insert(3, '3dT1 #Scans', df3d19.Sequence.value_counts())
df_nums.to_csv('C:\\Users\\kiril\\Thesis\\CoBra\\cobra\\data\\tables\\scan_tables\\scan_nums_2019.csv')
#%%
df19[(df19.Sequence=='dwi') & df19.MRAcquisitionType.isna()& (df19.NumberOfSlices>=64)]