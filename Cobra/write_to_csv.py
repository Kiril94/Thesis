# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 13:05:11 2021

@author: klein
"""

import os
import numpy as np
from data_access import load_data_tools as ld
import utilss.utils as utils
import pandas as pd
import importlib
import time
import pydicom
from utilss import basic
#importlib.reload(utils)
#importlib.reload(ld)

# In[Specify main directories]
base_data_dir = "Y:/"
out_pos_path = "Y:\\nii\\positive"
data_dirs = os.listdir(base_data_dir)
positive_dir = f"{base_data_dir}/positive" 
healthy_dirs = sorted([f"{base_data_dir}/{x}" for x \
                       in data_dirs if x.startswith('2019')])
csv_folder = "D:/Thesis/Cobra/tables"    


# In[delete last patient from ]
last_csv_file = "healthy_10_nn.csv"
last_csv_path = os.path.join(csv_folder, last_csv_file)

# In[get last patient value]
df_last = utils.load_scan_csv(last_csv_path)
last_patient_value = df_last['PatientID'].iloc[-1]
print(last_patient_value)

# In[delete last patient]
df_last = df_last[df_last.PatientID != last_patient_value]

# In[Create new csv]
df_last.to_csv(last_csv_path, index = False, header = True)
#print(df_last.iloc[-1:,2])

# In[get new last patient value]
df_last = utils.load_scan_csv(last_csv_path)
last_patient_value = df_last['PatientID'].iloc[-1]
print(last_patient_value)

# In[list patients from the last written patient]
subdir = healthy_dirs[9]
patient_list = sorted(utils.list_subdir(subdir))
print(subdir)
# In[get index]
last_pat_index = basic.get_index(patient_list, last_patient_value)
new_patient_list = patient_list[last_pat_index+1:]
print(last_patient_value)
print(patient_list[last_pat_index])

# In[Now write to csv]
utils.write_csv(last_csv_path, new_patient_list, 
                append=True)

# In[write positive to csv]
pos_patient_list = sorted(utils.list_subdir(positive_dir))      
csv_file = "pos_nn.csv"
csv_path = os.path.join(csv_folder, csv_file)
utils.write_csv(csv_path, pos_patient_list)

# In[Write neg to csv]
csv_columns = [x[0] for x in ld.get_scan_key_list()]
csv_folder = "D:/Thesis/Cobra/tables"
starting_month = 10
for month, subdir in enumerate(healthy_dirs[starting_month-1:starting_month]):
    print(f"converting files from {subdir}")
    csv_file = f"healthy_{month+starting_month}_nn.csv"
    #csv_file = f"test.csv"
    csv_path = os.path.join(csv_folder, csv_file)
    patient_list = sorted(utils.list_subdir(subdir))
    utils.write_csv(csv_path, patient_list)
    
# In[Create one csv for all 2019]
neg_tab_dirs = sorted([f"{csv_folder}/{x}" for x \
                       in os.listdir(csv_folder) if x.startswith('healthy') and 
                       not('_10_nn' in x)])  
df_neg_list = [utils.load_scan_csv(csv_path) for csv_path in \
               neg_tab_dirs]
df_all_neg_2019 = pd.concat(df_neg_list, axis=0, join="outer") 
# In[Write it to a csv]
df_all_neg_2019.to_csv(f"{csv_folder}/neg_all.csv", index = False, header = True)
