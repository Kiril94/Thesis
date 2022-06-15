# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 11:06:00 2021

@author: klein
"""
# %%
import pandas as pd
import matplotlib.pyplot as plt
import os, sys
from os.path import join, split
base_dir = split(os.getcwd())[0]
if base_dir not in sys.path:
    sys.path.append(base_dir)
import numpy as np
from stats_tools import vis as svis
from vis import mri_vis as mvis
import seaborn as sns
from utilities import stats, mri_stats
from utilities.basic import DotDict, p, sort_dict
import importlib
import pickle, gzip
import matplotlib as mpl
import matplotlib.lines as mlines
from collections import OrderedDict


from ast import literal_eval
plt.style.use(join(base_dir,'utilities', 'plot_style.txt'))
import importlib
import string
plt.style.use('ggplot')
#import proplot as pplt
import matplotlib.dates as mdates
import scipy.stats as ss

from pylab import cm
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
"""
params = {'figure.dpi':350,
        'legend.fontsize': 18,
        'figure.figsize': [8, 5],
         'axes.labelsize': 20,
         'axes.titlesize':20,
         'xtick.labelsize':20,
         'ytick.labelsize':20}
mpl.rcParams.update(params) 
plt.style.use('ggplot')
"""
#pplt.rc.cycle = 'ggplot'
colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] 
# %%
# In[tables directories]
fig_dir = f"{base_dir}/figs"
table_dir = f"{base_dir}/data/tables"
with gzip.open(join(table_dir, 'scan_3dt1_clean.gz'), 'rb') as f:
    df = pickle.load(f)

dfcm = pd.read_csv(join(base_dir, 'data\\t1_longitudinal\\imports\\dfscan_cov.csv'))
dfps = pd.read_csv(join(base_dir, 'data\\t1_longitudinal\\imports\\dfscan_ps.csv'))

dfcm = pd.merge(dfcm, df, how='left', on='SeriesInstanceUID')
dfps = pd.merge(dfps, df, how='left', on='SeriesInstanceUID')
MFS_k = 'MagneticFieldStrength'
M_k = 'Manufacturer'
MMN_k = 'ManufacturerModelName'
#%%
# Summarize manufacturers
man_dic = {'SIEMENS':'S', 'Philips Medical Systems':'P', 'Philips':'P',
        'Philips Healthcare':'P', 'GE MEDICAL SYSTEMS':'G','Agfa':'A'}
dfcm['man_new'] = dfcm[M_k].map(man_dic)
dfps['man_new'] = dfps[M_k].map(man_dic)

#%%
def count_change(df, key):
    df_case = df[df.case==1]
    df_con = df[df.case==0]
    case_gb = df_case.groupby('PatientID_x')[key].nunique()
    con_gb = df_con.groupby('PatientID_x')[key].nunique()
    case_change_frac = len(case_gb[case_gb==2])/len(case_gb)
    con_change_frac = len(con_gb[con_gb==2])/len(con_gb)
    return case_change_frac, con_change_frac
f_cm_ls = []
f_ps_ls = []
for key in [MFS_k, 'man_new', MMN_k]:
    f_cm_ls.append(count_change(dfcm, key))
    f_ps_ls.append(count_change(dfps, key))
#%%
fig, axs = plt.subplots(1,2)
case_label='case'
control_label='control'
for i, f_ps in zip(np.arange(0,7,3), f_ps_ls):
    if i>0:
        case_label=''
        control_label=''
    axs[0].barh(y = i, width = f_ps[0], height=1, color=colors[0],
        label=case_label)
    axs[0].barh(y = i+1, width = f_ps[1], height=1, color=colors[1],
        label=control_label)
    axs[0].set_title('PS', loc='center')
    
fig.legend()