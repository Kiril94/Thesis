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

# In[load positive csv]
table_all_dir = f"{table_dir}/all2019.csv"  
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

# In[Add sequence column and set it to one of the relevant values]
rel_keys = ['t1', 'gd', 't2_noflair', 't2s', 't2', 'swi', 'other']
rel_masks = [mask_dict[key] for key in rel_keys] 
df_all['Sequence'] = 0
for mask, key in zip(rel_masks, rel_keys):
    df_all['Sequence'][mask] = key
# In[check]
print(df_all[mask_dict.gd]['Sequence'])
# In[check]
print(df_all[SD_k][mask_dict.t2_flair])

# In[]

fig, ax = plt.subplots()
sns.scatterplot(x=TE_k, y=TR_k, hue='Sequence', 
                data=df_all[mask_dict.t1 | mask_dict.gd],
                ax=ax)
df_all[mask_dict.t1 | mask_dict.gd]

# In[Data]
print(f"all the identified sequences {mask_dict.relevant.sum()} out of {len(df_p)}")
print(f"nones or not identified in sequence column {mask_dict.none_nid.sum()}")
data = df_p[[TE_k, TR_k, IR_k, FA_k, SD_k]]
data_te_tr = data[mask_dict.none_nid].dropna(subset=[TE_k, TR_k])
data_te_tr_ti = data[mask_dict.none_nid].dropna(subset=[TE_k, TR_k, IR_k])
data_te_tr_fa = data[mask_dict.none_nid].dropna(subset=[TE_k, TR_k, FA_k])
print(f"Out of those nones or nid {len(data_te_tr)} have at least te and tr")
print(f"Out of those nones or nid {len(data_te_tr_fa)} have at least te, tr and fa")
print(f"Out of those nones or nid {len(data_te_tr_ti)} have at least te, tr and ti")
data_rel_te_tr = data[~mask_dict.none_nid].dropna(subset=[TE_k, TR_k])
data_rel_te_tr_ti = data[~mask_dict.none_nid].dropna(subset=[TE_k, TR_k, IR_k])
data_rel_te_tr_fa = data[~mask_dict.none_nid].dropna(subset=[TE_k, TR_k, FA_k])
print(f"Out of the identified {len(data_rel_te_tr)} have at least te and tr")
print(f"Out of the identified {len(data_rel_te_tr_fa)} have at least te, tr and fa")
print(f"Out of the identified {len(data_rel_te_tr_ti)} have at least te, tr and ti")

# In[]
data_other_none = data[mask_dict.other_none]
print(f"Volume that cannot be identified: {len(data_other_none)} out of\
      {len(data)}")
data_present = data[~mask_dict.other_none]
data_present = data_present.dropna(subset=[TE_k, TR_k, IR_k, FA_k])
data_missing = data[other_m | none_m]
data_missing = data_missing.dropna(subset=[TE_k, TR_k, IR_k, FA_k])
print(len(data_present))
print(len(data_missing))


# In[visualize sequences scatter]
fig, ax = plt.subplots(2,2,figsize=(10,10))
ax = ax.flatten()
sns.scatterplot(x=TE_k, y=TR_k, data=data_te_tr,ax=ax[0])
sns.scatterplot(x=TE_k, y=IR_k, legend=None,
                data=data_te_tr,
                ax=ax[1])
sns.scatterplot(x=IR_k, y=TR_k, legend=None,
                data=data_te_tr,
                ax=ax[2])
sns.scatterplot(x=IR_k, y=FA_k, legend=None,
                data=data_te_tr,
                ax=ax[3])
fig.suptitle('Non identified sequences (positive patients)', fontsize=20)
fig.tight_layout()
fig.savefig(f"{fig_dir}/pos/scatter_for_non_identified_sequences.png")
# In[]
sns.scatterplot(x=TE_k, y=IR_k, 
                hue='Sequence', data= df_p_clean,
                ax=ax[1])
sns.scatterplot(x=IR_k, y=TR_k, 
                hue='Sequence', data= df_p_clean,
                ax=ax[2])
sns.scatterplot(x=IR_k, y=FA_k, 
                hue='Sequence', data= df_p_clean,
                ax=ax[3])
plt.show()