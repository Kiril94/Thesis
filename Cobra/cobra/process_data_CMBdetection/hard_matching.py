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


#set matching parameters
categorical_matching_features = ['EchoTime','MagneticFieldStrength','SliceThickness']

#read tablespos
main_folder = "/home/neus/Documents/09.UCPH/MasterThesis/github/Thesis/Cobra/cobra"
tables_path = f'{main_folder}/tables'
pos_scan_info = pd.read_csv(f'{tables_path}/swi_pos_scans.csv')
neg_scan_info = pd.read_csv(f'{tables_path}/swi_neg_scans.csv')

#Create table to save info
matching_table = pd.DataFrame(columns=['ControlPatientID','CasePatientID'])
controls_per_case = []
#Hard matching
for idx,pos_scan in pos_scan_info.iterrows():
    case_id = pos_scan['PatientID']    
    matched_controls = neg_scan_info
    for feature in categorical_matching_features:
        matched_controls = matched_controls[ neg_scan_info[feature]==pos_scan[feature] ]
    
    controls_per_case.append(matched_controls.shape[0])
    matching = pd.DataFrame({'ControlPatientID': matched_controls['PatientID'],
                            'CasePatientID': case_id})
    matching_table = pd.concat([matching_table,matching])

#Save tables. 
folder_to_save = f'{tables_path}/SWIMatching/'
matching_table.sort_values('CasePatientID')
matching_table.to_csv(f'{folder_to_save}/swi_matched_control_case.csv',index=False)

#save into small tables
parts = np.array_split(matching_table,12)
for idx,part in enumerate(parts):
    part.to_csv(f'{folder_to_save}/swi_matched_control_case_{idx}.csv',index=False)

min_cpc,max_cpc,avg_cpc = np.min(controls_per_case),np.max(controls_per_case),np.average(controls_per_case)
print('*****************************************************')
print(f'CONTROLS PER CASE\nMin: {min_cpc:d}\tMax: {max_cpc:d}\tAverage: {avg_cpc:.3f}')

controls_per_case = np.array(controls_per_case)
print(f'N. case patients with no controls: {len(controls_per_case[controls_per_case<1])}')

print(f'N.cases before matching: {pos_scan_info.shape[0]} \t N.cases after matching: {matching_table["CasePatientID"].unique().shape[0]}')
print('*****************************************************')