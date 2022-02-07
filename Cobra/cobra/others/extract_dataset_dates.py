# -*- coding: utf-8 -*-
"""
Created on Mon Feb 7 12:06:00 2022

@author: neusRodeja
"""
import sys
sys.path.insert(0, '/home/neus/Documents/09.UCPH/MasterThesis/github/Thesis/Cobra/cobra/')

import numpy as np
import pandas as pd 
from utilities.utils import load_scan_csv,df_unique_values,find_n_slices,save_nscans
from stats_tools.vis import create_1d_hist,create_2d_hist,create_boxplot
import matplotlib.pyplot as plt 
from datetime import datetime as dt

main_folder = "/home/neus/Documents/09.UCPH/MasterThesis/github/Thesis/Cobra/cobra"
csv_folder = f"{main_folder}/tables"
csv_file_name = "neg_pos_clean"

table_all = load_scan_csv(f'{csv_folder}/{csv_file_name}.csv')

print(min(table_all['DateTime'].dropna()))
print(max(table_all['DateTime'].dropna()))