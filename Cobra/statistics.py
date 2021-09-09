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

# In[Define some helper functions]
def p(x):
    print(x)

def only_first_true(a, b):
    """takes two binary arrays
    and returns True
    where only the el. of the first array is true
    if only the second or both are true returns false"""
    return a&np.logical_not(a&b)

def mask_sequence_type(df, str_):
    """Checks in series description if it
    contains str_"""
    mask = df['SeriesDescription'].str.contains(str_, na=False)
    return mask

def check_sequence_flags(df, flags):
    """calls mask_sequence type for a list of flags and combines
    the masks with or"""
    masks = []
    for flag in flags:
        masks.append(mask_sequence_type(df, flag))
    mask = masks[0]
    for i in range(1, len(masks)):
        mask = mask | masks[i]
    return mask
# In[tables directories]
script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parent
table_dir = f"{base_dir}/tables"

# In[load positive csv]
pos_tab_dir = f"{table_dir}/pos.csv" 
df_p = pd.read_csv(pos_tab_dir, encoding= 'unicode_escape')
keys = df_p.keys()
p(keys)
# In[relevant interest]
gadolinium_f = ['dotarem', 'Gd','gd', 'GD', 'Dotarem', 
                    'Gadolinium', 'Dotarem']
t1_f = ['T1', 't1']
mprage_f = ['mprage', 'MPRAGE']
tfe_f = ['tfe', 'TFE']
fspgr_f = ['FSPGR']
flair_f = ['FLAIR','flair']
fse_f = ['FSE', 'fse']
gre_f  = ['GRE', 'gre']
t2s_f = ['T2\*']
dti_f = ['DTI', 'dti']
swi_f = ['SWI', 'swi']
# In[]
T1_mask = df_p['SeriesDescription'].str.contains('swi', na=False)
#FLAIR_mask = df_p['SeriesDescription'].str.contains('FLAIR', na=False)

print(df_p[T1_mask].SeriesDescription)

# In[Keys]

mprage_flags = []
t1_mask = df_p['SeriesDescription'].str.contains('t1', na=False)
print(df_p[t1_mask].SeriesDescription)

# In[Questions]
# whats FGRE
# 