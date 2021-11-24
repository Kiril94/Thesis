# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 10:52:01 2021

@author: klein

Convert dicom files to niftiis
"""
#%%
import access_sif_data.load_data_tools as ld
import os
from dcm2nii import dcm2nii
import time
import nibabel as nib
import datetime
import numpy as np
import matplotlib.pyplot as plt
from utilities import basic, utils

#%%
# In[main directories]
base_data_dir = "Y:/"
out_pos_path = "Y:\\nii\\positive"
data_dirs = os.listdir(base_data_dir)
positive_dir = f"{base_data_dir}/positive" 
healthy_dirs = sorted([f"{base_data_dir}/{x}" for x \
                       in data_dirs if x.startswith('2019')])
print(f"main directories: {data_dirs}")
#%% 
# In[Convert t1 sequences]
src_main_dir = "D:\Thesis\Cobra\cobra\\figs\Presentation\Study_design\dcm\\t1\series"
dest_main_dir = "D:\Thesis\Cobra\cobra\\figs\Presentation\Study_design\\nii\\t1"
for src_dir in basic.list_subdir(src_main_dir):
    dcm2nii.convert_dcm2nii(src_dir, dest_main_dir)
#%%
# In[Get all positive patients]
pos_patients_list = basic.list_subdir(positive_dir)
#%%
# In[Number of converted patients]
conv_patients_list = os.listdir(out_pos_path)
conv_patients = len(conv_patients_list)
print(conv_patients)

# In[non converted patients]
pos_patients_id = [os.path.split(dir_)[1] for dir_ in pos_patients_list]
non_conv_patients = set(pos_patients_id)- set(conv_patients_list)
print(f"non converted patients:{len(non_conv_patients)}")
non_conv_patients_dirs = sorted([os.path.join(positive_dir, non_conv_pat)\
                          for non_conv_pat in non_conv_patients])
# In[Convert positive patients]

start = time.time()
#patient_counter = len(non_conv_patients)
out_pos_path = "Y:\\nii\\positive\\test_tags"
for patient_dir in pos_patients_list[100:101]:
    patient_timer = time.time()
    patient = ld.Patient(patient_dir)
    patient_id = patient.get_id()
    scan_dirs = patient.get_scan_directories()
    out_patient_dir = os.path.join(out_pos_path, patient_id)
    if not os.path.exists(out_patient_dir):
        os.makedirs(out_patient_dir)
        print(f"{out_patient_dir} created")
    for scan_dir in scan_dirs[:1]:
        dcm2nii.convert_dcm2nii(scan_dir, out_patient_dir)
        print('|',end=(''))
    #patient_counter -= 1
    #print(patient_counter)
    print(str(datetime.datetime.now()))
stop = time.time()
print(f"The conversion took: {stop-start} s")


# In[Test compression]

conversion_times = []
patient = ld.Patient("D:/Thesis/Cobra/data/0b630d10621e9c5d831a8053f95125b6")
patient_id = patient.get_id()
scan_dirs = patient.get_scan_directories()
for compression in range(9,10):
    out_patient_dir = os.path.join(f"D:/Thesis/Cobra/data/{patient_id}_converted",
                                   str(compression))
    start = time.time()
    if not os.path.exists(out_patient_dir):
        os.makedirs(out_patient_dir)
        print(f"{out_patient_dir} created")    
    for scan_dir in scan_dirs:   
        dcm2nii.dcm2nii(scan_dir, out_patient_dir, compression=compression,
                            verbose=1)
    stop = time.time()
    conversion_times.append(stop-start)
    print(f"The conversion took: {stop-start} s")

# In[Compare sizes for different compression levels]

sizes = np.zeros(9)
gz_directory = "D:/Thesis/Cobra/data/0b630d10621e9c5d831a8053f95125b6_converted"
for i, compression in enumerate(range(1,10)):
    print(f"{i}, {compression}")
    sizes[i] = utils.get_size(os.path.join(gz_directory, str(compression)),
                              unit='')
    
# In[plot sizes]
conversion_times = np.array([50.8, 68.0,74.77, 80.65,73.595, 89.36,105.119,121.117, 134.9])
fig, ax1 = plt.subplots()
ax1.set_xlabel('Compression level')
ax1.plot(np.arange(1,10), sizes, color='r', label='size')
ax1.set_ylabel('size in MB')
ax2 = ax1.twinx()
ax2.plot(np.arange(1,10), conversion_times, label='conversion time')
ax1.legend()
ax2.legend()
ax2.set_ylabel('time in s')
# In[Compare access]

out_patient_dir = os.path.join("D:/Thesis/Cobra/data/0b630d10621e9c5d831a8053f95125b6_converted")
access_times = np.zeros(9)
for i, compression in enumerate(range(1,10)):
    test_compression_dir = f"{out_patient_dir}/{compression}"
    files = sorted(utils.list_subdir(test_compression_dir))
    start = time.time()
    for file in files:
        if file.endswith('.nii.gz'):
            img_mat = nib.load(file)
        else:
            print("not nii")
    stop = time.time()   
    access_times[i] = stop-start
    print(f"For compression={compression} the access took: {stop-start} s")

