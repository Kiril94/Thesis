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