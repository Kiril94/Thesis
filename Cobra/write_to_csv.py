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
importlib.reload(utils)
importlib.reload(ld)


# In[Specify main directories]
base_data_dir = "Y:/"
out_pos_path = "Y:\\nii\\positive"
data_dirs = os.listdir(base_data_dir)
positive_dir = f"{base_data_dir}/positive" 
healthy_dirs = sorted([f"{base_data_dir}/{x}" for x \
                       in data_dirs if x.startswith('2019')])
csv_folder = "D:/Thesis/Cobra/tables"    

# In[delete last patient from ]
last_csv_file = "healthy_9.csv"
last_csv_path = os.path.join(csv_folder, last_csv_file)

# In[write df to csv without last patient]
df_last = pd.read_csv(last_csv_path,encoding= 'unicode_escape')
last_patient_key = df_last['PatientID'].keys()[-1]
last_patient_value = df_last['PatientID'][last_patient_key]
last_patient_mask = ~(df_last['PatientID']==last_patient_value)
df_new = df_last[last_patient_mask]
#df_new.to_csv(last_csv_path, index=False)
num_patients_written = len(df_new['PatientID'].unique())
print(last_patient_value)
# In[list patients from the last written patient]
subdir = healthy_dirs[8]
patient_list = sorted(utils.list_subdir(subdir))[num_patients_written:]

# In[write from the last written patient]

csv_path_10 = os.path.join(csv_folder, "healthy_10_n.csv")
df_last = pd.read_csv(csv_path_10, encoding= 'unicode_escape')
num_patients = len(df_last['PatientID'].unique())

# In[]
pat_counter = 0
patient_list = sorted(utils.list_subdir(healthy_dirs[9]))[num_patients:]
utils.write_csv(csv_path_10, patient_list)

# In[Select patients]
patient_list = sorted(utils.list_subdir(positive_dir))    

# In[Get Repetition Times only]
RT_list = []
start = time.time()
for pat in patient_list:
    scan_directories = ld.Patient(pat).get_scan_directories()
    for scan_dir in scan_directories:
        try:
            dicom_file_dir = os.path.join(scan_dir, os.listdir(scan_dir)[0])
            dicom = pydicom.dcmread(dicom_file_dir)
            try:
                RT_list.append(float(dicom.RepetitionTime))
            except:
                RT_list.append(dicom.RepetitionTime)
        except:
            if len(os.listdir(scan_dir))==0:
                print(f"{scan_dir} is empty")
                continue
        print('.', end='')
    print(f"Patient {pat} stored")
stop = time.time()
print(f"the conversion took {stop-start}")


# In[write positive to csv]
csv_file = "pos_n.csv"
csv_path = os.path.join(csv_folder, csv_file)
utils.write_csv(csv_path, patient_list)

# In[Write neg to csv]
csv_columns = [x[0] for x in ld.get_scan_key_list()]
csv_folder = "D:/Thesis/Cobra/tables"
starting_month = 11
for month, subdir in enumerate(healthy_dirs[starting_month-1:]):
    print(f"converting files from {subdir}")
    csv_file = f"healthy_{month+starting_month}_nn.csv"
    #csv_file = f"test.csv"
    csv_path = os.path.join(csv_folder, csv_file)
    patient_list = sorted(utils.list_subdir(subdir))
    utils.write_csv(csv_path, patient_list)
    
# In[]
a = str('[1,2]')
print(str(a))

