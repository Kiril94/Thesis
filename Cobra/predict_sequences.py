# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 15:50:50 2021

@author: klein
"""
#import xgboost
import os
from pathlib import Path
import pandas as pd
#from utilss import stats

from utilss import utils
from utilss import mri_stats
from utilss.basic import DotDict
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# In[tables directories]
script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parent
fig_dir = f"{base_dir}/figs/basic_stats"
table_dir = f"{base_dir}/tables"
fig_dir = f"{base_dir}/figs"

# In[Define useful keys]
TE_k = 'EchoTime'
TR_k = 'RepetitionTime'
TI_k = 'InversionTime'
FA_k = 'FlipAngle'
SD_k = 'SeriesDescription'
PID_k = 'PatientID'
time_k = 'InstanceCreationTime'
date_k = 'InstanceCreationDate'
DT_k = 'DateTime'
SID_k = 'SeriesInstanceUID'
SS_k = 'ScanningSequence'
SV_k = 'SequenceVariant'
SN_k = 'SequenceName'
SO_k = 'ScanOptions'
ETL_k = 'EchoTrainLength'

# In[load all csv]
rel_cols = [SID_k, SD_k, TE_k, TR_k, FA_k, TI_k, ETL_k, SS_k, SV_k, SN_k, PID_k]
table_all_dir = f"{table_dir}/neg_pos.csv"  
df_all = utils.load_scan_csv(table_all_dir)[rel_cols]

# In[Select only relevant columns]
print(f"all elements {len(df_all)}")
df_all = df_all[rel_cols].dropna(subset=[SID_k, PID_k, SS_k, SV_k, TR_k])
print(f"after dropping nans {len(df_all)}")

# In[Turn ScanningSequence into multi-hot encoded]
s = df_all[SS_k].explode()
columns_list = list(df_all.columns)
columns_list.remove(SS_k)
df_all = df_all[columns_list].join(pd.crosstab(s.index, s))
del s
# In[Turn SequenceVariant into multi-hot encoded]
s = df_all[SV_k].explode()
columns_list = list(df_all.columns)
columns_list.remove(SV_k)
df_all = df_all[columns_list].join(pd.crosstab(s.index, s))
del s

# In[Turn sparse columns into sparse arrays]
columns_list = list(df_all.columns)
sparse_columns = ['EP', 'GR', 'IR', 'RM', 'SE', 'DE', 'MP', 'MTC', 
                  'OSP', 'SK', 'SP', 'SS', 'TOF']
for item in sparse_columns:
    columns_list.remove(item)

df_all = utils.convert_to_sparse_pandas(
    df_all, columns_list)
print(df_all.dtypes)

# In[Define tags]
mask_dict, tag_dict = mri_stats.get_masks_dict(df_all)

# In[Count number of relevant patients in 2019]
print(len(df_2019[mask_dict.t1 | mask_dict.t2 | mask_dict.t2s | mask_dict.t1gd \
        | mask_dict.t2gd | mask_dict.gd | mask_dict.swi | mask_dict.flair \
        | mask_dict.dwi][PID_k].unique()))
print(len(df_2019[PID_k].unique()))

# In[Add sequence column and set it to one of the relevant values]
sq = 'Sequence'
rel_keys = ['t1', 't1gd', 't2', 't2gd', 't2s', 'swi', 'flair','none_nid', 
            'gd', 'dwi',]
rel_masks = [mask_dict[key] for key in rel_keys] 
df_all[sq] = "other"
for mask, key in zip(rel_masks, rel_keys):
    df_all['Sequence'][mask] = key
df_all[sq][mask_dict.t1gd] = "t1"
df_all[sq][mask_dict.t2gd] = "t2"
df_all[sq][mask_dict.gd] = "none_nid"
sequences = ['t1', 't2', 't2s', 'flair','swi', 'dwi', 'other', 'none_nid', ]
# In[Count number of volumes with a specific sequence]
seq_count = df_all['Sequence'].value_counts()
print(seq_count)
# In[visualize number of volumes sequences]
vis.bar_plot(seq_count.keys(), seq_count.values, figsize=(13,6), xlabel='Sequence',
             xtickparams_ls=16, save_plot=True, title='All Patients',
             figname=f"{fig_dir}/sequence_pred/sequences_count.png")

# In[visualize sequences scatter]
fig, ax = plt.subplots(2,2,figsize=(10,10))
ax = ax.flatten()
sns.scatterplot(x=TE_k, y=TR_k, data=df_all, hue='Sequence',ax=ax[0])
sns.scatterplot(x=TE_k, y=IR_k, legend=None,hue='Sequence',
                data=df_all,
                ax=ax[1])
sns.scatterplot(x=IR_k, y=TR_k, legend=None,hue='Sequence',
                data=df_all,
                ax=ax[2])
sns.scatterplot(x=IR_k, y=FA_k, legend=None,hue='Sequence',
                data=df_all,
                ax=ax[3])
fig.suptitle('All Sequences', fontsize=20)
fig.tight_layout()
fig.savefig(f"{fig_dir}/sequence_pred/scatter_for_all.png")


# In[Relevant columns for prediction]

seq_vars = [SID_k, SD_k, TE_k, TR_k, FA_k, TI_k, ETL_k, SS_k, SV_k, SN_k]
df_all2 = df_all[seq_vars]
# In[tets]
print(df_all.keys())

# In[]
# In[encode inputs]
# Look at a subset of the dataframe 
mlb = MultiLabelBinarizer(sparse_output=True)

df_all_mh = df_all2.join(
            pd.DataFrame.sparse.from_spmatrix(
                mlb.fit_transform(df_all2.pop(SS_k)),
                index=df_all2.index,
                columns=mlb.classes_))
print(df_all_mh)
# In[]
mlb = MultiLabelBinarizer(sparse_output=True)
print(mlb.fit_transform(df_all2.pop(SS_k)))
