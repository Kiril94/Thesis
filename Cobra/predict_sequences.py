# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 15:50:50 2021

@author: klein
"""
import xgboost
import os
from pathlib import Path
import pandas as pd
from utilss import stats
import ast
from utilss import utils
from utilss import mri_stats
from utilss.basic import DotDict
import seaborn as sns
import matplotlib.pyplot as plt


# In[tables directories]
script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parent
fig_dir = f"{base_dir}/figs/basic_stats"
table_dir = f"{base_dir}/tables"
fig_dir = f"{base_dir}/figs"

# In[load all csv]
table_all_dir = f"{table_dir}/neg_pos.csv"  
df_all = utils.load_scan_csv(table_all_dir)

# In[Define useful keys]
TE_k = 'EchoTime'
TR_k = 'RepetitionTime'
IR_k = 'InversionTime'
FA_k = 'FlipAngle'
SD_k = 'SeriesDescription'
PID_k = 'PatientID'
time_k = 'InstanceCreationTime'
date_k = 'InstanceCreationDate'

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

# In[tets]
print(df_all.keys())

# In[encode inputs]
# Look at a subset of the dataframe 
multi_hot = df_all[:100]['SequenceType'].pivot_table(
    index=None, aggfunc=[len], fill_value=0)
