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
from stats_tools import vis as svis
from glob import iglob
from vis import vis


# In[Define some helper functions]
def p(x):
    print(x)

def only_first_true(a, b):
    """takes two binary arrays
    and returns True
    where only the el. of the first array is true
    if only the second or both are true returns false"""
    return a&np.logical_not(a&b)

def mask_sequence_type(df, str_, key='SeriesDescription'):
    """Checks in series description if it
    contains str_"""
    mask = df[key].str.contains(str_, na=False)
    return mask

def check_sequence_tags(df, tags, key='SeriesDescription'):
    """calls mask_sequence type for a list of tags and combines
    the masks with or"""
    masks = []
    for tag in tags:
        masks.append(mask_sequence_type(df, tag, key))
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
fig_dir = f"{base_dir}/figs/basic_stats"
table_dir = f"{base_dir}/tables"

# In[load positive csv]
pos_tab_dir = f"{table_dir}/pos.csv" 
df_p = pd.read_csv(pos_tab_dir, encoding= 'unicode_escape')
keys = df_p.keys()
p(keys)

# In[Get number of scans per patient]
scans_per_patient = df_p.groupby('PatientID').size()
figure = svis.nice_histogram(
    scans_per_patient, np.arange(1,110,2), 
    show_plot=True, xlabel='# scans per patient',
    save=True, figname = f"{fig_dir}/pos/scans_per_patient.png")

# In[Sort scans by manufacturer]
manufactureres = df_p['Manufacturer'].unique()
p(manufactureres)
philips_t = ['Philips Healthcare', 'Philips Medical Systems',
             'Philips'] 
philips_c = check_sequence_tags(df_p, philips_t, 'Manufacturer').sum()
siemens_c = mask_sequence_type(df_p, 'SIEMENS', 'Manufacturer').sum()
gms_c = mask_sequence_type(df_p, 'GE MEDICAL SYSTEMS', 'Manufacturer').sum()
agfa_c = mask_sequence_type(df_p, 'Agfa', 'Manufacturer').sum()
none_c = df_p['Manufacturer'].isnull().sum()

# In[visualize scanner manufacturer counts]

fig, ax = plt.subplots(1,figsize = (10,6))
manufacturers_unq = ['Philips', 'SIEMENS', 'GEMS', 'Agfa', 'none']
counts = np.array([philips_c, siemens_c, gms_c, agfa_c, none_c])
vis.bar_plot(manufacturers_unq, counts, xlabel='Manufacturer', 
             save_plot=True, figname=f"{fig_dir}/pos/manufacturers_count.png")

# In[Model Name]
philips_m = check_sequence_tags(df_p, philips_t, 'Manufacturer')
philips_models = df_p[philips_m]['ManufacturerModelName'].unique()
#mft_model = df_p['ManufacturerModelName'].unique()
p(mft_model)
p(philips_models)
#p(keys)
# In[relevant tags]
t1_t = ['T1', 't1']
mpr_t = ['mprage', 'MPRAGE', 'mpr', 'MPR']
tfe_t = ['tfe', 'TFE']
spgr_t = ['FSPGR']

flair_t = ['FLAIR','flair']

t2_t = ['T2', 't2']
fse_t = ['FSE', 'fse', '']

t2s_t = ['T2\*', 't2\*']
gre_t  = ['GRE', 'gre']

dti_t = ['DTI', 'dti']

swi_t = ['SWI', 'swi']
dwi_t = ['DWI', 'dwi']
gd_t = ['dotarem', 'Dotarem', 'Gd','gd', 'GD', 'Gadolinium']

# In[Get corresponding masks]

t1_m = check_sequence_tags(df_p, t1_t)
mpr_m = check_sequence_tags(df_p, mpr_t)
tfe_m = check_sequence_tags(df_p, tfe_t)
spgr_m = check_sequence_tags(df_p, spgr_t)

flair_m = check_sequence_tags(df_p, flair_t)

mpr_m = check_sequence_tags(df_p, mpr_t)
fse_m = check_sequence_tags(df_p, fse_t)

t2s_m = check_sequence_tags(df_p, t2s_t)
gre_m  = check_sequence_tags(df_p, gre_t)

dti_m = check_sequence_tags(df_p, dti_t)

swi_m = check_sequence_tags(df_p, swi_t)
dwi_m = check_sequence_tags(df_p, dwi_t)
gd_m = check_sequence_tags(df_p, gd_t)

# In[Count flags]
t1_m = check_sequence_flags(df_p, t1_t)
t2_m = check_sequence_flags(df_p, t2_t)
t1_c = np.count_nonzero(check_sequence_flags(df_p, t1_t))
t1_mprage_c = t1_c
print(t1_c)

# In[tests]
T1_mask = df_p['SeriesDescription'].str.contains('dyes', na=False)
#FLAIR_mask = df_p['SeriesDescription'].str.contains('FLAIR', na=False)
print(df_p[T1_mask].SeriesDescription)
# In[Questions]
# whats FGRE
# 