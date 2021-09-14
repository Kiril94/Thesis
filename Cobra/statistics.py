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
from utilss import utils
from data_access import load_data_tools as ld
import pydicom
from utilss.utils import dotdict
import utilss.utils as utils
from dotmap import DotMap
import importlib
importlib.reload(utils)

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
df_p = utils.load_scan_csv(pos_tab_dir)
keys = df_p.keys()
p(keys)
p(f"Number of patients = {len(df_p.PatientID.unique())}")
# In[Usefule keys]
TE_k = 'EchoTime'
TR_k = 'RepetitionTime'
IR_k = 'InversionTime'
FA_k = 'FlipAngle'
SD_k = 'SeriesDescription'
PID_k = 'PatientID'
time_k = 'InstanceCreationTime'
date_k = 'InstanceCreationDate'

# In[Convert time and date to datetime for efficient access]
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
print(f"The nunmber of studies is {sum(num_studies_l)}")
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


# In[Sequence Types, define tags]
tag_dict = {}
tag_dict['t1'] = ['T1', 't1']
tag_dict['mpr'] = ['mprage', 'MPRAGE']
print('MPRAGE is always T1w')
tag_dict['tfe'] = ['tfe', 'TFE']
tag_dict['spgr'] = ['FSPGR']
print("The smartbrain protocol occurs only for philips")
tag_dict['smartbrain'] = ['SmartBrain']

tag_dict['flair'] = ['FLAIR','flair', 'Flair']

tag_dict['t2'] = ['T2', 't2']
tag_dict['fse'] = ['FSE', 'fse', '']

tag_dict['t2s'] = ['T2\*', 't2\*']
tag_dict['gre']  = ['GRE', 'gre']

tag_dict['dti']= ['DTI', 'dti']
print("There is one perfusion weighted image (PWI)")
tag_dict['swi'] = ['SWI', 'swi']
tag_dict['dwi'] = ['DWI', 'dwi']
tag_dict['adc'] = ['ADC', 'Apparent Diffusion Coefficient']
tag_dict['gd'] = ['dotarem', 'Dotarem', 'Gd','gd', 'GD', 'Gadolinium']
tag_dict['stir'] = ['STIR']
tag_dict['tracew'] = ['TRACEW']
tag_dict['asl'] = ['ASL']
tag_dict['cest'] = ['CEST']
tag_dict['survey'] = ['SURVEY', 'Survey', 'survey']
tag_dict['angio'] = ['TOF', 'ToF', 'tof','angio', 'Angio', 'ANGIO', 'SWAN']
print("TOF:time of flight angriography, SWAN: susceptibility-weighted angiography")
tag_dict = dotdict(tag_dict)
# Look up: MIP (maximum intensity projection), SmartBrain, 
# TOF (time of flight angriography), ADC?, STIR (Short Tau Inversion Recovery),
# angio, Dynamic Contrast-Enhanced Magnetic Resonance Imaging (DCE-MRI) 
# In[Get corresponding masks]
# take mprage to the t1
mask_dict = dotdict({key : stats.check_tags(df_p, tag) for key, tag in tag_dict.items()})
#mprage is always t1 https://pubmed.ncbi.nlm.nih.gov/1535892/
mask_dict['t1'] = stats.check_tags(df_p, tag_dict.t1) \
    | stats.check_tags(df_p, tag_dict.mpr)
mask_dict['t1tfe'] = mask_dict.t1 & mask_dict.tfe
mask_dict['t1spgr'] = mask_dict.t1 & mask_dict.spgr
mask_dict['t2_flair'] = stats.only_first_true(
    stats.check_tags(df_p, tag_dict.t2), mask_dict.t2s)
mask_dict['t2_noflair'] = stats.only_first_true(mask_dict.t2_flair, mask_dict.flair)# real t2
mask_dict.none = df_p['SeriesDescription'].isnull()

print("we are interested in t1, t2_noflair, flair, swi, dwi, dti, angio")
print("combine all masks with an or and take complement")
mask_dict.all = mask_dict.t1 | mask_dict.flair | mask_dict.t2_noflair \
    | mask_dict.t2s | mask_dict.dwi | mask_dict.dti | mask_dict.swi \
        | mask_dict.angio | mask_dict.adc | mask_dict.stir \
            |mask_dict.survey | mask_dict.none
mask_dict.other = ~mask_dict.all
# In[Look at 'other' group] combine all the relevant masks to get others

p(df_p[mask_dict.other].SeriesDescription)
other_seq_series = df_p[mask_dict.other].SeriesDescription
other_seq_series_sort = other_seq_series.sort_values(axis=0, ascending=True).unique()
pd.DataFrame(other_seq_series_sort).to_csv(f"{base_dir}/tables/other_sequences.csv")


# In[Get counts]
counts_dict = dotdict({key : mask.sum() for key, mask in mask_dict.items()})
print(counts_dict)

# In[visualize basic sequences]
sequences_names = ['T1+\nMPRAGE', 'T2', 'FLAIR', 'T2*', 'SWI', 'DWI', 
                   'angio', 'ADC', 'survey','TRACEW', 'Other','None']
seq_counts = np.array([counts_dict.t1, counts_dict.t2_noflair, counts_dict.flair,
                       counts_dict.t2s, counts_dict.swi, 
                       counts_dict.dwi, counts_dict.angio, 
                       counts_dict.adc, counts_dict.survey, counts_dict.tracew,
                       counts_dict.other, 
                       counts_dict.none])
vis.bar_plot(sequences_names, seq_counts, figsize=(13,6), xlabel='Sequence',
             xtickparams_ls=16, save_plot=True, title='Positive Patients',
             figname=f"{fig_dir}/pos/basic_sequences_count.png")

# In[Look at the distributions of TE and TR for different seq]

df_p.loc[mask_dict.t1, 'Sequence'] = 'T1'
df_p.loc[mask_dict.t2_noflair,'Sequence'] = 'T2'
df_p.loc[mask_dict.t2s,'Sequence'] = 'T2S'
df_p.loc[mask_dict.flair,'Sequence'] = 'FLAIR'
df_p_clean = df_p.dropna(subset=[TE_k, TR_k])

# In[visualize sequences scatter]
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

# In[Extract dates from series description if not present in InstanceCreationData]

#p(df_p['InstanceCreationDate'].dropna())
p(f"number of scans without date {df_p['InstanceCreationDate'].isnull().sum()}\
  out of {len(df_p)}")
date_mask = df_p['SeriesDescription'].str.contains('2020', na=False)
#p(df_p[date_mask]['SeriesDescription'].count())
# these are not that many

# In[Search for combinations of FLAIR, SWI, T1]
gb_pat = df_p.groupby(PID_k)
#grouped masks followd by
flair_m = stats.check_tags(gb_pat, flair_t).groupby(PID_k).any()
swi_m = stats.check_tags(gb_pat, swi_t).groupby(PID_k).any()
t1_m = stats.check_tags(gb_pat, t1_t).groupby(PID_k).any()
t2_m = stats.check_tags(gb_pat, t2_t).groupby(PID_k).any()

flair_swi_t1_m = flair_m & swi_m & t1_m 
p(f"{flair_swi_t1_m.sum()} patients have\
  the sequences flair, swi and t1")
flair_swi_t1_t2_m = flair_m & swi_m & t1_m & t2_m
p(f"{flair_swi_t1_t2_m.sum()} patients have\
  the sequences flair, swi,t1 and t2")
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