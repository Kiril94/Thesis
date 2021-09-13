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
import pydicom
importlib.reload(ld)


# In[Specify main directories]
base_data_dir = "Y:/"
out_pos_path = "Y:\\nii\\positive"
data_dirs = os.listdir(base_data_dir)
positive_dir = f"{base_data_dir}/positive" 
healthy_dirs = sorted([f"{base_data_dir}/{x}" for x \
                       in data_dirs if x.startswith('2019')])
    
# In[Specify csv path]
csv_folder = "D:/Thesis/Cobra/tables"
csv_file = "pos_n.csv"
csv_path = os.path.join(csv_folder, csv_file)
csv_columns = [x[0] for x in ld.get_scan_key_list()]



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
csv_columns = [x[0] for x in ld.get_scan_key_list()]
subdir = healthy_dirs[8]
patient_list = sorted(utils.list_subdir(subdir))[num_patients_written:]

# In[write from the last written patient]
with open(last_csv_path, 'a', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    start = time.time()
    for pat in patient_list:
        scan_directories = ld.Patient(pat).get_scan_directories()
        for scan_dir in scan_directories[10:]:
            try:
                data = ld.get_scan_dictionary(scan_dir, reconstruct_3d=False)
            except:
                print(f"Sleep for 5s, maybe connection is lost, check that dir is not empty")
                time.sleep(5)
                if len(os.listdir(scan_dir))==0:
                    print(f"{scan_dir} is empty")
                    continue
                data = ld.get_scan_dictionary(scan_dir, reconstruct_3d=False)
            try:
                writer.writerow(data)  
            except IOError:
                print("I/O error")
            print('.', end='')
        print(f"{pat} stored to csv")
    stop = time.time()
    print(f"the conversion took {stop-start}")





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



# In[]
df_pos = pd.read_csv(csv_path, encoding= 'unicode_escape')
df_pos.rename(columns={'RepititionTime': 'RepetitionTime'},
          inplace=True, errors='raise')
#df_pos['RepetitionTime'] = RT_list
#print(df_pos['RepititionTime'])
print(len(RT_list))
print(len(df_pos))
# In[write positive to csv]

csv_columns = [x[0] for x in ld.get_scan_key_list()]
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    writer.writeheader()
    start = time.time()
    pat_counter = 0
    for pat in patient_list:
        scan_directories = ld.Patient(pat).get_scan_directories()
        for scan_dir in scan_directories:
            try:
                data = ld.get_scan_dictionary(scan_dir, reconstruct_3d=False)
            except:
                print(f"Sleep for 5s, director {scan_dir} not found")
                time.sleep(5)
                if len(os.listdir(scan_dir))==0:
                    print(f"{scan_dir} is empty")
                    continue
                data = ld.get_scan_dictionary(scan_dir, reconstruct_3d=False)
            try:
                writer.writerow(data)
            except IOError:
                print("I/O error")
            print('.', end='')
        pat_counter+=1
        if pat_counter%100==0:
            print(f"{pat_counter} patients written")
            print(f"time passed {time.time()-start}")
        print(f"{pat} stored to csv")
        
    stop = time.time()
print(f"the conversion took {stop-start}")


# In[Write neg to csv]
csv_columns = [x[0] for x in ld.get_scan_key_list()]
csv_folder = "D:/Thesis/Cobra/tables"
for month, subdir in enumerate(healthy_dirs[9:]):
    print(f"converting files from {subdir}")
    csv_file = f"healthy_{month+10}_n.csv"
    csv_path = os.path.join(csv_folder, csv_file)
    patient_list = sorted(utils.list_subdir(subdir))
    pat_counter = 0
    with open(csv_path, 'w', newline='') as csvfile:
        start = time.time()
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()    
        for pat in patient_list:
            print(f"Writing {os.path.split(pat)[1]} to csv")
            scan_directories = ld.Patient(pat).get_scan_directories()
            for scan_dir in scan_directories:
                try:
                    data = ld.get_scan_dictionary(scan_dir, reconstruct_3d=False)
                except:
                    print("Sleep for 5s, maybe connection is lost, check that dir is not empty")
                    time.sleep(5)
                    if len(os.listdir(scan_dir))==0:
                        print(f"{scan_dir} is empty")
                        continue
                    data = ld.get_scan_dictionary(scan_dir, reconstruct_3d=False)
                try:
                    writer.writerow(data)
                except IOError:
                    print("I/O error")
                print('.', end='')
            pat_counter+=1
            if pat_counter%100==0:
                print(f"{pat_counter} patients written")
                print(f"time passed: {time.time()-start}")
        stop = time.time()
        print(f"the conversion took {stop-start}")
    print(f"all patients in {subdir} converted")
# In[]
df = pd.read_csv(csv_path)
print(df['EchoTime'])
# In[]
print(os.path.split("D:/as/as")[1])