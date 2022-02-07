# -*- coding: utf-8 -*-
"""
Created on Thu Feb 3 12:06:00 2022

@author: neusRodeja
"""
import sys
sys.path.insert(0, '/home/neus/Documents/09.UCPH/MasterThesis/github/Thesis/Cobra/cobra/')

import numpy as np
import pandas as pd 
from utilities.utils import load_scan_csv,df_unique_values,find_n_slices,save_nscans
from stats_tools.vis import create_1d_hist,create_2d_hist,create_boxplot
import matplotlib.pyplot as plt 

main_folder = "/home/neus/Documents/09.UCPH/MasterThesis/github/Thesis/Cobra/cobra"
figs_folder = f'{main_folder}/figs/cmb_stats'
csv_folder = f"{main_folder}/tables"
csv_ids_name = "mri_patients_ids"
csv_file_name = "neg_pos_clean"

all_scans = pd.read_csv(f'{csv_folder}/{csv_file_name}.csv')
all_scans.drop_duplicates(subset='PatientID',inplace=True)

all_scans['PatientID'].to_csv(f'{csv_folder}/{csv_ids_name}.csv',index=False)