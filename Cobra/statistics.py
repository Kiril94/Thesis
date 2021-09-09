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

def check_sequence_tags(df, flags):
    """calls mask_sequence type for a list of tags and combines
    the masks with or"""
    masks = []
    for tag in tags:
        masks.append(mask_sequence_type(df, tag))
    mask = masks[0]
    for i in range(1, len(masks)):
        mask = mask | masks[i]
    return mask
# In[Test functions]
masks = [np.array([1, 0, 0]), np.array([0, 0, 0]), np.array([1, 1, 0])]
mask = masks[0]
for i in range(1, len(masks)):
    mask = mask | masks[i]
print(mask)
# In[tables directories]
script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parent
table_dir = f"{base_dir}/tables"

# In[load positive csv]
pos_tab_dir = f"{table_dir}/pos.csv" 
df_p = pd.read_csv(pos_tab_dir, encoding= 'unicode_escape')
keys = df_p.keys()
p(keys)

# In[relevant tags]
t1_t = ['T1', 't1']
mprage_t = ['mprage', 'MPRAGE', 'mpr', 'MPR']
tfe_t = ['tfe', 'TFE']
fspgr_t = ['FSPGR']

flair_t = ['FLAIR','flair']

t2_t = ['T2', 't2']
fse_t = ['FSE', 'fse', '']

t2s_t = ['T2\*', 't2\*']
gre_t  = ['GRE', 'gre']

dti_t = ['DTI', 'dti']

swi_t = ['SWI', 'swi']
dwi_t = ['DWI', 'dwi']
gadolinium_t = ['dotarem', 'Dotarem', 'Gd','gd', 'GD', 'Gadolinium']

# In[Get corresponding masks]

# In[Count flags]
t1_m = check_sequence_flags(df_p, t1_t)
t2_m = check_sequence_flags(df_p, t2_t)
t1_c = np.count_nonzero(check_sequence_flags(df_p, t1_t))
t1_mprage_c = t1_c
print(t1_c)

# In[tests]
T1_mask = df_p['SeriesDescription'].str.contains('\+gd', na=False)
#FLAIR_mask = df_p['SeriesDescription'].str.contains('FLAIR', na=False)
print(df_p[T1_mask].SeriesDescription)
# In[Questions]
# whats FGRE
# 