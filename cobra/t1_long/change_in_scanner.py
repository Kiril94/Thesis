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
fig, axs = plt.subplots(figsize=(7,3))
case_label='cases'
control_label_cm='CM controls'
control_label_ps='PSM controls'
for i, f_ps, f_cm in zip(np.arange(0,11,4), f_ps_ls, f_cm_ls):
    if i>0:
        case_label=''
        control_label_cm=''
        control_label_ps=''
    print(f_ps, 'ps')
    print(f_cm, 'cm')
    axs.barh(y = i, width = f_ps[0], height=1, color=colors[0],
        label=case_label)
    bp = axs.barh(y = i+1, width = f_ps[1], height=1, color=colors[1],
        label=control_label_ps, hatch='x')
    axs.barh(y = i+2, width = f_cm[1], height=1, color=colors[1],
        label=control_label_cm, hatch='|')
axs.set_yticks([1, 5, 9])
axs.set_yticklabels([r'$B_0$', 'Manufacturer', 'Model Name'])
axs.set_xlabel( 'Change of scanner from baseline to follow-up scan in ' +r'$N/N_{\mathrm{tot}}$', ha='center', fontsize=14)
axs.legend(fontsize=12, bbox_to_anchor=(0.8, 1.1), ncol=3)
fig.savefig(join(fig_dir, 'longitudinal', 'change.png'),
     dpi=300, bbox_inches='tight')
