# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 13:05:11 2021

@author: klein
"""

import os
import csv
import numpy as np
from data_access import load_data_tools as ld
import utilss.utils as utils
import pandas as pd
import importlib
import time
importlib.reload(ld)
# In[Specify main directories]
base_data_dir = "Z:/"
out_pos_path = "Z:\\nii\\positive"
data_dirs = os.listdir(base_data_dir)
positive_dir = f"{base_data_dir}/positive" 
healthy_dirs = sorted([f"{base_data_dir}/{x}" for x \
                       in data_dirs if x.startswith('2019')])
# In[Specify csv path]
csv_folder = "D:/Thesis/Cobra/tables"
csv_file = "pos.csv"
csv_path = os.path.join(csv_folder, csv_file)
csv_columns = [x[0] for x in ld.get_scan_key_list()]
# In[Select patients]
patient_list = sorted(utils.list_subdir(positive_dir))
# In[not converted patients]
pos_patients_id = [os.path.split(dir_)[1] for dir_ in patient_list]
df = pd.read_csv(csv_path)
stored_patient_ids = set(df.PatientID)
non_conv_patients = set(pos_patients_id) - set(stored_patient_ids)
print(f"non converted patients:{len(non_conv_patients)}")
patient_list = sorted([os.path.join(positive_dir, non_conv_pat)\
                          for non_conv_pat in non_conv_patients])
    
# In[Convert files]
with open(csv_path, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    writer.writeheader()
    start = time.time()
    for pat in patient_list:
        scan_directories = ld.Patient(pat).get_scan_directories()
        for scan_dir in scan_directories:
            data = ld.get_scan_dictionary(scan_dir, reconstruct_3d=False)
            try:
                writer.writerow(data)
            except IOError:
                print("I/O error")
        print('|', end='')
    stop = time.time()
print(f"the conversion took {stop-start}")
# In[]
df = pd.read_csv(csv_path)
print(df.keys())
# In[]
