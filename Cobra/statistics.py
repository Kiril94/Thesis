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
import datetime
import time
import seaborn as sns
from utilss import stats

# In[Define some helper functions]
def p(x):
    print(x)

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
pos_tab_dir = f"{table_dir}/pos_n.csv" 
df_p = pd.read_csv(pos_tab_dir, encoding= 'unicode_escape')
keys = df_p.keys()
p(keys)

# In[Convert time and date to datetime for efficient access]
time_k = 'InstanceCreationTime'
date_k = 'InstanceCreationDate'
df_p['DateTime'] = df_p[date_k] + ' ' +  df_p[time_k]
date_time_m = ~df_p['DateTime'].isnull()
df_p['DateTime'] = pd.to_datetime(df_p['DateTime'], format='%Y%m%d %H:%M:%S')



# In[Sort the the scans by time and count those that are less than 2 hours apart]
df_p_sorted = df_p.groupby('PatientID').apply(
    lambda x: (x.sort_values(by=['DateTime'], ascending=True)))
# In[Count the number of studies]
start = time.time()
patient_ids = df_p_sorted['PatientID'].unique()
num_studies_l = []
for patient in patient_ids:
    patient_mask = df_p_sorted['PatientID']==patient
    date_times = df_p_sorted[patient_mask]['DateTime']
    date_time0 = date_times[0]
    study_counter = 1
    for date_time in date_times[1:]:
        try:
            time_diff = date_time-date_time0
            if time_diff.total_seconds()/3600>2:
                study_counter += 1
                date_time0 = date_time
            else:
                pass
        except:
            print('NaT')
    num_studies_l.append(study_counter)
stop = time.time()

# In[Store the results]
ppatient_df = pd.DataFrame({'PatientId':[], 'NumStudies':[]})#storing results
ppatient_df['PatientID'] = patient_ids
ppatient_df['NumStudies'] =  num_studies_l   

# In[Show distribution of the studies]
num_studies_a = np.array(num_studies_l)
max_studies = max(num_studies_a)
svis.nice_histogram(num_studies_a, np.arange(.5, max_studies+.5),
                    show_plot=True, xlabel='Number of studies',
                    save=True, title='Positive Patients',
                    figname=f"{fig_dir}/pos/num_studies.png")


# In[Get number of acquired volumes per patient]
scans_per_patient = df_p.groupby('PatientID').size()
figure = svis.nice_histogram(
    scans_per_patient, np.arange(1,110,2), 
    show_plot=True, xlabel='# volumes per patient',
    save=True, figname = f"{fig_dir}/pos/volumes_per_patient.png",
    title='Positive Patients')


# In[Sort scans by manufacturer]
manufactureres = df_p['Manufacturer'].unique()
p(manufactureres)
philips_t = ['Philips Healthcare', 'Philips Medical Systems',
             'Philips'] 
philips_c = stats.check_tags(df_p, philips_t, 'Manufacturer').sum()
siemens_c = stats.mask_sequence_type(df_p, 'SIEMENS', 'Manufacturer').sum()
gms_c = stats.mask_sequence_type(df_p, 'GE MEDICAL SYSTEMS', 'Manufacturer').sum()
agfa_c = stats.mask_sequence_type(df_p, 'Agfa', 'Manufacturer').sum()
none_c = df_p['Manufacturer'].isnull().sum()

# In[visualize scanner manufacturer counts]
fig, ax = plt.subplots(1,figsize = (10,6))
manufacturers_unq = ['Philips', 'SIEMENS', 'GEMS', 'Agfa', 'none']
counts = np.array([philips_c, siemens_c, gms_c, agfa_c, none_c])
vis.bar_plot(manufacturers_unq, counts, xlabel='Manufacturer', 
             save_plot=True, figname=f"{fig_dir}/pos/manufacturers_count.png",
             title='Positive Patients')


# In[Model Name]
philips_m = stats.check_tags(df_p, philips_t, 'Manufacturer')
siemens_m = stats.mask_sequence_type(df_p, 'SIEMENS', 'Manufacturer')
gms_m = stats.mask_sequence_type(df_p, 'GE MEDICAL SYSTEMS', 'Manufacturer')

model_k = 'ManufacturerModelName'
philips_models_vc = df_p[philips_m][model_k].value_counts().to_dict()
siemens_models_vc = df_p[siemens_m][model_k].value_counts().to_dict()
gms_models_vc = df_p[gms_m][model_k].value_counts().to_dict()

# In[summarize small groups]
philips_models_vc_new = stats.group_small(philips_models_vc, 1000)
siemens_models_vc_new = stats.group_small(siemens_models_vc, 200)
gms_models_vc_new = stats.group_small(gms_models_vc, 200)

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

fig.suptitle('Positive Patients', fontsize=20)
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
t1_m = stats.check_tags(df_p, t1_t) | stats.check_tags(df_p, mpr_t)

mpr_m = stats.check_tags(df_p, mpr_t)
t1mpr_m = t1_m & mpr_m
tfe_m = stats.check_tags(df_p, tfe_t)
t1tfe_m = t1_m & tfe_m
spgr_m = stats.check_tags(df_p, spgr_t)
t1spgr_m = t1_m & spgr_m

flair_m = stats.check_tags(df_p, flair_t)

fse_m = stats.check_tags(df_p, fse_t)

t2s_m = stats.check_tags(df_p, t2s_t)
gre_m  = stats.check_tags(df_p, gre_t)

dwi_m = stats.check_tags(df_p, dwi_t)
gd_m = stats.check_tags(df_p, gd_t)

t2_flair_m = stats.only_first_true(stats.check_tags(df_p, t2_t), t2s_m)
t2_noflair_m = stats.only_first_true(t2_flair_m, flair_m)# real t2
dti_m = stats.check_tags(df_p, dti_t)

swi_m = stats.check_tags(df_p, swi_t) 

angio_m = stats.check_tags(df_p, angio_t)
smartbrain_m  = stats.check_tags(df_p, smartbrain_t)

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
             xtickparams_ls=18, save_plot=True, title='Positive Patients',
             figname=f"{fig_dir}/pos/basic_sequences_count.png")


# In[Does smartbrain occer only for certain scaners?]
p(df_p[smartbrain_m].Manufacturer.unique())
# Yes only for philips
p(keys)


# In[Look at the distributions of TE and TR for different seq]
TE_k = 'EchoTime'
TR_k = 'RepetitionTime'
IR_k = 'InversionTime'
FA_k = 'FlipAngle'
df_p.loc[t1_m, 'Sequence'] = 'T1'
df_p.loc[t2_noflair_m,'Sequence'] = 'T2'
df_p.loc[t2s_m,'Sequence'] = 'T2S'
df_p.loc[flair_m,'Sequence'] = 'FLAIR'
df_p_clean = df_p.dropna(subset=[TE_k, TR_k])

print(df_p.Sequence.dropna())
# In[]
print(df_p_clean[TR_k])

# In[]
fig, ax = plt.subplots(2,2,figsize=(10,10))
ax = ax.flatten()
sns.scatterplot(x=TE_k, y=TR_k, 
                hue='Sequence', data=df_p_clean,ax=ax[0])
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
# In[tests]
T1_mask = df_p['SeriesDescription'].str.contains('MP2RAGE', na=False)
#FLAIR_mask = df_p['SeriesDescription'].str.contains('FLAIR', na=False)
print(df_p[T1_mask].SeriesDescription)
# In[Extract dates from series description if not present in InstanceCreationData]

#p(df_p['InstanceCreationDate'].dropna())
p(f"number of scans without date {df_p['InstanceCreationDate'].isnull().sum()}\
  out of {len(df_p)}")
date_mask = df_p['SeriesDescription'].str.contains('2020', na=False)
#p(df_p[date_mask]['SeriesDescription'].count())
# these are not that many
# In[]

# In[Search for combinations of FLAIR, SWI, T1]

flair_swi_t1_m = flair_m | swi_m | t1_m
p(f"{len(df_p[flair_swi_t1_m]['PatientID'].unique())} patients have\
  the sequences flair, swi and t1")


# In[when where the scans performed]
scan_months = np.array([int(date[5:7]) for date in df_p['InstanceCreationDate'].dropna()])
svis.nice_histogram(scan_months, np.arange(.5,13.5), show_plot=(True), 
                    xlabel='month', save=(True), title='Number of acquired volumes for positive patients',
                    figname=f"{fig_dir}/pos/scan_months.png" )
# Notes: time and data are sometimes given in the series description
#p(scan_months[10])
# In[Check number os studies per patient]

p(df_p.groupby(['PatientID', 'InstanceCreationDate', 'InstanceCreationTime']).count())#['SeriesInstanceUID'])
#svis.nice_histogram(scan_months, np.arange(.5,13.5), show_plot=(True), 
#                    xlabel='month', save=(True), 
#                    figname=f"{fig_dir}/pos/scan_months.png" )