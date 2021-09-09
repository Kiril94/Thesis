# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 11:06:00 2021

@author: klein
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np
def p(x):
    print(x)

def only_first_true(a, b):
    """takes two binary arrays
    and returns only elements as True
    where only the el. of the first array is true"""
    return a&np.logical_not(a&b)
# In[tables directories]
script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parent
table_dir = f"{base_dir}/tables"

# In[load positive csv]
pos_tab_dir = f"{table_dir}/pos.csv" 
df_p = pd.read_csv(pos_tab_dir, encoding= 'unicode_escape')
keys = df_p.keys()
# In[relevant interest]
gadolinium_f = ['dotarem', 'Gd','gd', 'GD', 'Dotarem', 
                    'Gadolinium', 'Dotarem']
t1_f = ['T1', 't1']
mprage_f = ['mprage', 'MPRAGE']
tfe_f = ['tfe', 'TFE']
fspgr_f = ['FSPGR']

# In[]
T1_mask = np.array(
    df_p['SeriesDescription'].str.contains('IR', na=False))
FLAIR_mask = np.array(
    df_p['SeriesDescription'].str.contains('FLAIR', na=False))
print(T1_mask & FLAIR_mask)
print(df_p[T1_mask].SeriesDescription)
# In[]
a = np.array([1, 0, 1, 0])
b = np.array([1, 0, 0, 1])
print(a&np.logical_not(a&b))

# In[Keys]

mprage_flags = []
t1_mask = df_p['SeriesDescription'].str.contains('t1', na=False)
print(df_p[t1_mask].SeriesDescription)