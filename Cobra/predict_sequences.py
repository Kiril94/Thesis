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
from utilss.basic import DotDict
import seaborn as sns
import matplotlib.pyplot as plt


# In[tables directories]
script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parent
fig_dir = f"{base_dir}/figs/basic_stats"
table_dir = f"{base_dir}/tables"

# In[load positive csv]
pos_tab_dir = f"{table_dir}/pos_n.csv"  
df_p = utils.load_scan_csv(pos_tab_dir)

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
tag_dict = {}
tag_dict['t1'] = ['T1', 't1']
tag_dict['mpr'] = ['mprage', 'MPRAGE']
print('MPRAGE is always T1w')
tag_dict['tfe'] = ['tfe', 'TFE']
tag_dict['spgr'] = ['FSPGR']
print("The smartbrain protocol occurs only for philips")
tag_dict['smartbrain'] = ['SmartBrain']

tag_dict['flair'] = ['FLAIR','flair', 'Flair']

tag_dict['t2'] = ['T2', 't2']
tag_dict['fse'] = ['FSE', 'fse']

tag_dict['t2s'] = ['T2\*', 't2\*']
tag_dict['gre']  = ['GRE', 'gre']

tag_dict['dti']= ['DTI', 'dti']
print("There is one perfusion weighted image (PWI)")
tag_dict['swi'] = ['SWI', 'swi']
tag_dict['dwi'] = ['DWI', 'dwi']
tag_dict['adc'] = ['ADC', 'Apparent Diffusion Coefficient']
tag_dict['gd'] = ['dotarem', 'Dotarem', 'Gd','gd', 'GD', 'Gadolinium']
tag_dict['stir'] = ['STIR']
tag_dict['tracew'] = ['TRACEW']
tag_dict['asl'] = ['ASL']
tag_dict['cest'] = ['CEST']
tag_dict['survey'] = ['SURVEY', 'Survey', 'survey']
tag_dict['angio'] = ['TOF', 'ToF', 'tof','angio', 'Angio', 'ANGIO', 'SWAN']
print("TOF:time of flight angriography, SWAN: susceptibility-weighted angiography")
tag_dict = DotDict(tag_dict)
# In[Get corresponding masks]
# take mprage to the t1
mask_dict = DotDict({key : stats.check_tags(df_p, tag) for key, tag in tag_dict.items()})
#mprage is always t1 https://pubmed.ncbi.nlm.nih.gov/1535892/
mask_dict['t1'] = stats.check_tags(df_p, tag_dict.t1) \
    | stats.check_tags(df_p, tag_dict.mpr)
mask_dict['t1tfe'] = mask_dict.t1 & mask_dict.tfe
mask_dict['t1spgr'] = mask_dict.t1 & mask_dict.spgr
mask_dict['t2_flair'] = stats.only_first_true(
    stats.check_tags(df_p, tag_dict.t2), mask_dict.t2s)
mask_dict['t2_noflair'] = stats.only_first_true(mask_dict.t2_flair, mask_dict.flair)# real t2
mask_dict.fse_only = stats.only_first_true(
    stats.only_first_true(mask_dict.fse, mask_dict.t1), mask_dict.t2)
print("we are interested in t1, t2_noflair, flair, swi, dwi, dti, angio")
print("combine all masks with an or and take complement")

mask_identified = mask_dict.t1
for mask in mask_dict.values():
    mask_identified = mask_identified | mask
mask_dict.identified = mask_identified

mask_dict.relevant = mask_dict.t1 | mask_dict.flair | mask_dict.t2_noflair \
    | mask_dict.t2s | mask_dict.dwi | mask_dict.swi \
        | mask_dict.angio | mask_dict.adc 

mask_dict.none = df_p['SeriesDescription'].isnull()       
mask_dict.none_nid = mask_dict.none | ~mask_dict.identified #either non or not identified
mask_dict.other = ~mask_dict.none_nid & ~mask_dict.relevant# nont none identified and non relevant


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