# -*- coding: utf-8 -*-
"""
Created on Mon Nov  15 12:06:00 2021

@author: neusRodeja
"""
import sys
sys.path.insert(0, '/home/neus/Documents/09.UCPH/MasterThesis/github/Thesis/Cobra/cobra/')

import numpy as np
import pandas as pd 
from utilities.utils import load_scan_csv,df_unique_values,find_n_slices,save_nscans
from stats_tools.vis import create_1d_hist,create_2d_hist,create_boxplot
import matplotlib.pyplot as plt 

label = 'swi_pos'
main_folder = "/home/neus/Documents/09.UCPH/MasterThesis/github/Thesis/Cobra/cobra"
figs_folder = f'{main_folder}/figs/cmb_stats'
csv_folder = f"{main_folder}/tables"
csv_swi_file_name = f"{label}_scans"
csv_file_name = "neg_pos_clean"

table_all = load_scan_csv(f'{csv_folder}/{csv_file_name}.csv')
###### Inspect quality for SWI positive scans.
mask_nan_values =  (table_all['InstanceCreationDate'].notna()&table_all['DateTime'].notna())
swi_pos_scans = table_all[(table_all['Positive']==1)&(table_all['Sequence']=='swi')&(table_all['days_since_test']>-3)&mask_nan_values]
#swi_pos_patients = swi_pos_scans.drop_duplicates(substet=['PatientID'])

n_scans = swi_pos_scans.shape[0]
n_patients = swi_pos_scans.drop_duplicates(subset='PatientID').shape[0]
print(f'Number of scans:\t{n_scans}\nNumber of patients:\t{n_patients}')

scans_groupedbyPatient = swi_pos_scans.groupby(['PatientID'])
# Scans per patient distribution
scans_perPatient = scans_groupedbyPatient.size()
scans_min,scans_max = np.min(scans_perPatient),np.max(scans_perPatient)
fig,ax = plt.subplots()
ax.set(ylabel='Counts',xlabel='#Scans per patient')
hist_data = create_1d_hist(ax,scans_perPatient,14,(-0.5,13.5),'Positive patients with SWI scans',display_counts=True)
fig.savefig(f'{figs_folder}/swi_scans_per_patient.png')

#Take patients that have only 1,2 or 3 scans
patients_id = scans_perPatient[scans_perPatient<4].keys()
swi_pos_scans = swi_pos_scans[swi_pos_scans['PatientID'].isin(patients_id)]
swi_pos_scans = swi_pos_scans.sort_values('days_since_test',ascending=False)

#Days since test for patients with multiple scans
swi_pos_scans = swi_pos_scans.groupby('PatientID').first().reset_index() ###TAKING THE LATEST SCAN 
days = swi_pos_scans['days_since_test']
days_min,days_max = np.min(days),np.max(days)
fig,ax = plt.subplots()
ax.set(ylabel='Counts',xlabel='#Days since positive test')
create_1d_hist(ax,days,100,(days_min,days_max),'Postive patients with multiple SWI')
fig.savefig(f'{figs_folder}/swi_days_since_test.png')

##Drop row with nan values 
column_subset = ['SliceThickness','SpacingBetweenSlices','PixelSpacing','MagneticFieldStrength','Rows','Columns','DateTime']
swi_pos_scans.dropna(subset=column_subset)

#Save the number of slices per scan and save the table
swi_pos_scans = save_nscans(swi_pos_scans,f'{csv_folder}/{csv_swi_file_name}.csv')

#Save extra files to send to computerome
swi_pos_scans['PatientID'].to_csv(f'{csv_folder}/{label}_patientIds.csv',index=False)
swi_pos_scans['DateTime'].to_csv(f'{csv_folder}/{label}_dateTime.csv',index=False)


