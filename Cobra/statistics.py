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
import itertools


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

def check_tags(df, tags, key='SeriesDescription'):
    """calls mask_sequence type for a list of tags and combines
    the masks with or"""
    masks = []
    for tag in tags:
        masks.append(mask_sequence_type(df, tag, key))
    mask = masks[0]
    for i in range(1, len(masks)):
        mask = mask | masks[i]
    return mask

def group_small(dict_, threshold, keyword='other'):
    newdic={}
    for key, group in itertools.groupby(dict_, lambda k: keyword \
                                        if (dict_[k]<threshold) else k):
         newdic[key] = sum([dict_[k] for k in list(group)]) 
    return newdic

    
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
philips_c = check_tags(df_p, philips_t, 'Manufacturer').sum()
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

philips_m = check_tags(df_p, philips_t, 'Manufacturer')
siemens_m = mask_sequence_type(df_p, 'SIEMENS', 'Manufacturer')
gms_m = mask_sequence_type(df_p, 'GE MEDICAL SYSTEMS', 'Manufacturer')

philips_models_vc = df_p[philips_m]['ManufacturerModelName'].value_counts().to_dict()
siemens_models_vc = df_p[siemens_m]['ManufacturerModelName'].value_counts().to_dict()
gms_models_vc = df_p[gms_m]['ManufacturerModelName'].value_counts().to_dict()

# In[summarize small groups]
philips_models_vc_new = group_small(philips_models_vc, 1000)
siemens_models_vc_new = group_small(siemens_models_vc, 200)
gms_models_vc_new = group_small(gms_models_vc, 200)

# In[visualize]
fig, ax = plt.subplots(2,2, figsize=(10,10))
ax = ax.flatten()

lbls_ph = philips_models_vc_new.keys()
szs_ph = philips_models_vc_new.values()
lbls_si = siemens_models_vc_new.keys()
szs_si = siemens_models_vc_new.values()
lbls_gm = gms_models_vc_new.keys()
szs_gm = gms_models_vc_new.values()

ax[0].pie(szs_ph,  labels=lbls_ph, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax[0].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax[0].set_title('Philips', fontsize=20)
ax[1].pie(szs_si,  labels=lbls_si, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax[1].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax[1].set_title('Siemens', fontsize=20)
ax[2].pie(szs_gm,  labels=lbls_gm, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax[2].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax[2].set_title('GMS', fontsize=20)
ax[-1].axis('off')

fig.tight_layout()
plt.subplots_adjust(wspace=.5, hspace=None)
plt.show()
fig.savefig(f"{fig_dir}/pos/model_name_pie_chart.png")




# In[Sequence Types]
t1_t = ['T1', 't1']
mpr_t = ['mprage', 'MPRAGE']
tfe_t = ['tfe', 'TFE']
spgr_t = ['FSPGR']
smartbrain_t = ['SmartBrain']

flair_t = ['FLAIR','flair']

t2_t = ['T2', 't2']
fse_t = ['FSE', 'fse', '']

t2s_t = ['T2\*', 't2\*']
gre_t  = ['GRE', 'gre']

dti_t = ['DTI', 'dti']

swi_t = ['SWI', 'swi']
dwi_t = ['DWI', 'dwi']
gd_t = ['dotarem', 'Dotarem', 'Gd','gd', 'GD', 'Gadolinium']

angio_t = ['TOF', 'ToF', 'angio', 'Angio', 'ANGIO']
# Look up: MIP (maximum intensity projection), SmartBrain, 
# TOF (time of flight angriography), ADC?, STIR (Short Tau Inversion Recovery),
# angio, Dynamic Contrast-Enhanced Magnetic Resonance Imaging (DCE-MRI) 
# In[Get corresponding masks]
# take mprage to the t1
t1_m = check_tags(df_p, t1_t) | check_tags(df_p, mpr_t)

mpr_m = check_tags(df_p, mpr_t)
t1mpr_m = t1_m & mpr_m
tfe_m = check_tags(df_p, tfe_t)
t1tfe_m = t1_m & tfe_m
spgr_m = check_tags(df_p, spgr_t)
t1spgr_m = t1_m & spgr_m

flair_m = check_tags(df_p, flair_t)

fse_m = check_tags(df_p, fse_t)

t2s_m = check_tags(df_p, t2s_t)
gre_m  = check_tags(df_p, gre_t)

dwi_m = check_tags(df_p, dwi_t)
gd_m = check_tags(df_p, gd_t)

t2_flair_m = only_first_true(check_tags(df_p, t2_t), t2s_m)
t2_noflair_m = only_first_true(t2_flair_m, flair_m)# real t2
dti_m = check_tags(df_p, dti_t)

swi_m = check_tags(df_p, swi_t) 

angio_m = check_tags(df_p, angio_t)
smartbrain_m  = check_tags(df_p, smartbrain_t)

none_m = df_p['SeriesDescription'].isnull()
# we are interested in t1, t2_noflair, flair, swi, dwi, dti
# combine all masks with an or and take complement
all_m = t1_m | flair_m | t2_noflair_m | t2s_m | dwi_m | dti_m | swi_m | angio_m | none_m 
other_m = ~all_m
# In[Look at 'other' group] combine all the relevant masks to get others
p(df_p[other_m].SeriesDescription)
other_seq_series = df_p[other_m].SeriesDescription
other_seq_series_sort = other_seq_series.sort_values(axis=0, ascending=True).unique()
pd.DataFrame(other_seq_series_sort).to_csv(f"{base_dir}/tables/other_sequences.csv")


# In[Get counts]

t1mpr_c = t1mpr_m.sum()
t1tfe_c = t1tfe_m.sum()
t1spgr_c = t1spgr_m.sum()

# counts we are interested in
flair_c = flair_m.sum()
t1_c = t1_m.sum()
t2_c = t2_flair_m.sum()
t2noflair_c = t2_noflair_m.sum()
t2s_c = t2s_m.sum()
dti_c = dti_m.sum()
swi_c = swi_m.sum()
dwi_c = dwi_m.sum()
angio_c = angio_m.sum()

none_c = none_m.sum()
other_c = other_m.sum()

# In[visualize basic sequences]
sequences_basic = ['T1+MPR', 'T2', 'FLAIR', 'T2*', 'DTI', 'SWI', 'DWI', 'angio',
                   'Other',
                   'None']
seq_counts = np.array([t1_c, t2noflair_c, flair_c, t2s_c, 
                       dti_c, swi_c, dwi_c, angio_c, other_c, none_c])
vis.bar_plot(sequences_basic, seq_counts, figsize=(13,6), xlabel='Sequence',
             xtickparams_ls=18, save_plot=True, 
             figname=f"{fig_dir}/pos/basic_sequences_count.png")


# In[Does smartbrain occer only for certain scaners?]
p(df_p[smartbrain_m].Manufacturer.unique())
# Yes only for philips



# In[tests]
T1_mask = df_p['SeriesDescription'].str.contains('MP2RAGE', na=False)
#FLAIR_mask = df_p['SeriesDescription'].str.contains('FLAIR', na=False)
print(df_p[T1_mask].SeriesDescription)
# In[Extract dates from series description if not present in InstanceCreationData]

p(keys)
p(df_p['InstanceCreationDate'].dropna())
p(df_p['InstanceCreationDate'].isnull().count())
date_mask = df_p['SeriesDescription'].str.contains('2020', na=False)
p(df_p[date_mask]['SeriesDescription'].count())
# these are not that many

# In[Search for combinations of FLAIR, SWI, T1]

flair_swi_t1_m = flair_m | swi_m | t1_m
p(df_p[flair_swi_t1_m].groupby(['PatientID']).count())

# In[when where the scans performed]
scan_months = np.array([int(date[5:7]) for date in df_p['InstanceCreationDate'].dropna()])
svis.nice_histogram(scan_months, np.arange(.5,13.5), show_plot=(True), 
                    xlabel='month', save=(True), 
                    figname=f"{fig_dir}/pos/scan_months.png" )
# Notes: time and data are sometimes given in the series description
#p(scan_months[10])
# In[Check number os studies per patient]

p(df_p.groupby(['PatientID', 'InstanceCreationDate', 'InstanceCreationTime']).count())#['SeriesInstanceUID'])
#svis.nice_histogram(scan_months, np.arange(.5,13.5), show_plot=(True), 
#                    xlabel='month', save=(True), 
#                    figname=f"{fig_dir}/pos/scan_months.png" )