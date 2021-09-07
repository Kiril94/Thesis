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
# In[Select patients]
patient_list = sorted(utils.list_subdir(positive_dir))

# In[Convert files]
csv_folder = "D:/Thesis/Cobra/tables"
csv_file = "test.csv"
csv_path = os.path.join(csv_folder, csv_file)
csv_columns = [x[0] for x in ld.get_scan_key_list()]

with open(csv_path, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    writer.writeheader()
    start = time.time()
    for pat in patient_list[:10]:
        scan_directories = ld.Patient(pat).get_scan_directories()
        for scan_dir in scan_directories:
            data = ld.get_scan_dictionary(scan_dir, reconstruct_3d=False)
            try:
                writer.writerow(data)
            except IOError:
                print("I/O error")
    stop = time.time()
print(f"the conversion took {stop-start}")
# In[]
df = pd.read_csv(csv_path)
print(df.InstanceCreationDate)
# In[]
print(9*26000/60/60)
