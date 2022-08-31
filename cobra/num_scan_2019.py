#%%
import pandas as pd
import matplotlib.pyplot as plt
from sympy import ordered
df = pd.read_pickle("C:\\Users\\kiril\\Thesis\\CoBra\\cobra\\data\\tables\\scan_tables\\scan_after_sq_pred_dst_nos_date.pkl")

#%%
df = df[['PatientID','InstanceCreationDate','Sequence', 'MRAcquisitionType']]
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
df.InstanceCreationDate.isna().sum()