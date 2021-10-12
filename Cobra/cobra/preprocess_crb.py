# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 13:50:34 2021

@author: klein
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
from os.path import join
from pathlib import PurePath as Path
import numpy as np
from stats_tools import vis as svis
from vis import mri_vis as mvis
import seaborn as sns
from utilities import stats, utils, mri_stats, basic
from utilities.basic import DotDict, p, sort_dict
import importlib

# In[Usefule keys]
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
MFS_k = 'MagneticFieldStrength'

# In[tables directories]
script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parent
fig_dir = f"{base_dir}/figs/basic_stats"
table_dir = f"{base_dir}/tables"
all_tab_dir = f"{table_dir}/neg_pos.csv"
# In[load df]
df_all = utils.load_scan_csv(all_tab_dir)
mask_dict, tag_dict = mri_stats.get_masks_dict(df_all)

# In[T2* GRE ]
mask_t2s = df_all.Sequence == 't2s' 
mask_t2s_gre = mask_t2s & mask_dict['gre']
print(f"{mask_dict['t2s'].sum()} T2* sequences")
print(f"{mask_t2s_gre.sum()} T2* GRE sequences")
df_all.Sequence = np.where(mask_t2s_gre, 't2s_gre', df_all.Sequence)
mask_t2s_gre = df_all.Sequence == 't2s_gre' 
print(df_all[mask_t2s_gre].SeriesDescription.unique())

# In[FLAIR CE]
mask_flair = df_all.Sequence == 'flair'
mask_gd = mask_dict['gd']
mask_flair_ce = mask_flair & mask_gd
df_all.Sequence = np.where(mask_flair_ce, 'flair_ce', df_all.Sequence)
mask_flair_new = df_all.Sequence=='flair'
print(df_all[mask_flair_new].SeriesDescription.unique())
# In[SWI MIP]
mask_swi = df_all.Sequence == 'swi'
mask_mip = mask_dict['mip']
mask_swi_mip = mask_swi & mask_mip
df_all.Sequence = np.where(mask_swi_mip, 'swi_mip', df_all.Sequence)
mask_swi_new = df_all.Sequence=='swi'
print(df_all[mask_swi_new].SeriesDescription.unique())

# In[Save new dataframe]
#df_all.to_csv(all_tab_dir, index=False)














