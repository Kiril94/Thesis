"""
Created on Mon Aug 30 10:32:43 2021
@author: klein
"""

import data_access.load_data_tools as ld
import os
from glob import iglob
import importlib
from utilss import utils
import time
from utilss import dicom2nifti
import json
from vis import vis
import nibabel as nib
import datetime
import pydicom
import numpy as np
import matplotlib.pyplot as plt
importlib.reload(ld)
importlib.reload(vis)
importlib.reload(dicom2nifti)
importlib.reload(utils)
def p(string): print(string)

# In[main directories]
base_data_dir = "Y:/"
out_pos_path = "Y:\\nii\\positive"
data_dirs = os.listdir(base_data_dir)
positive_dir = f"{base_data_dir}/positive" 
healthy_dirs = sorted([f"{base_data_dir}/{x}" for x \
                       in data_dirs if x.startswith('2019')])
print(f"main directories: {data_dirs}")

# In[Number of converted patients]
conv_patients_list = os.listdir(out_pos_path)
conv_patients = len(conv_patients_list)
print(conv_patients)

# In[Get all positive patients]
pos_patients_list = utils.list_subdir(positive_dir)

# In[Read some dcm files]

#patient = ld.Patient(pos_patients_list[100])
#scan_dirs = patient.get_scan_directories()
scan_path = utils.list_subdir("D:\\Thesis\\Cobra\\data\\test_compression\\0b630d10621e9c5d831a8053f95125b6\\5274cbd4b01b48a67071a35e252a692c\\MR\\26b082d69057e5884eb3ac0634966629")
dicom = pydicom.dcmread(scan_path[0])
with open("D:/Thesis/Cobra/dicom.txt", "w") as f:
    f.write(str(dicom))

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
        dicom2nifti.dcm2nii(scan_dir, out_patient_dir)
        print('|',end=(''))
    #patient_counter -= 1
    print(patient_counter)
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
        dicom2nifti.dcm2nii(scan_dir, out_patient_dir, compression=compression,
                            verbose=1)
    stop = time.time()
    times.append(stop-start)
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

out_patient_dir = os.path.join(f"D:/Thesis/Cobra/data/0b630d10621e9c5d831a8053f95125b6_converted")
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
# In[]
out_patient_dir = os.path.join(f"D:\Thesis\Cobra\data\0b630d10621e9c5d831a8053f95125b6_converted")
# In[look at converted patients]
con_pat_paths = [os.path.join(out_pos_path, conv_pat)\
                 for conv_pat in conv_patients_list]
converted_patient = con_pat_paths[400]
files = utils.list_subdir(converted_patient)
print(files[:2])
# read json header
json_file = files[]
with open(json_file) as f:
  header = json.load(f)
print(header)
#img_mat = nib.load(files[1])
#data = img_mat.get_fdata()
#print(f"the data shape is: {data.shape}")
#vis.display3d(data, axis=2, start_slice=10, num_slices=20)
# In[Count scans number]
scan_counters = {}
for healthy_dir in healthy_dirs[8:9]:
    print(f"counting studies in {healthy_dir}")    
    patient_list = utils.list_subdir(healthy_dir)
    scan_counter = 0
    for pat_dir in patient_list:
        print('|',end=(''))
        scan_counter += len(ld.Patient(pat_dir).get_scan_directories())
    scan_counters[healthy_dir] = scan_counter
    print(f'number of scans in {healthy_dir} =  {scan_counter}')
# In[Count study number]
study_counters = {}
for pos_dir in [positive_dir]:
    print(f"counting studies in {positive_dir}")
    patient_list = utils.list_subdir(pos_dir)
    study_counter = 0
    for pat_dir in patient_list:
        print('|',end=(''))
        study_counter += sum(1 for _ in iglob(pat_dir))
    study_counters[pos_dir] = study_counter
    print(f'number of studies in {pos_dir} =  {study_counter}')
# In[]
patient_list = utils.list_subdir(positive_dir)
# In[count studies]
study_counter = 0
for pat_dir in patient_list[:3]:
    print('|',end=(''))
    print(pat_dir)
    print([f for f in os.listdir(pat_dir)])
    study_counter += sum(1 for _ in iglob(pat_dir))
study_counters[pos_dir] = study_counter
print(f'number of studies in {pos_dir} =  {study_counter}')
# In[Count number of documented studies]
report_counters = {}
for dir_ in healthy_dirs[4:5]:
    patient_list = utils.list_subdir(dir_)
    report_counter = 0
    report_counter += sum(1 for _ in iglob(f"{dir_}/*/*/DOC/*/*.pdf"))
    print('|',end=(''))
    report_counters[dir_] = report_counter 
    print(f'number of study reports in {dir_} = {report_counter}')

# In[Count Patients]
count = []
for subdir in healthy_dirs[:8]:
    healthy_count = sum([1 for _ in os.listdir(subdir)])
    count.append(healthy_count)
    print(f"subdir: {subdir}, contains: {healthy_count}")
# In[]
print(healthy_dirs[11:12])
#print(sum([1,1,1]))
# In[]

# number of pos patients = 831
# number of neg patients = 24908
# 2019_01, contains: 2567
# 2019_02, contains: 2252
# 2019_03, contains: 2186
# 2019_04, contains: 2297
# 2019_05, contains: 2397
# 2019_06, contains: 2250
# 2019_07, contains: 1746
# 2019_09, contains: 2392
# 2019_10, contains: 2424
# 2019_10, contains: 2472
# 2019_11, contains: 2472
# 2019_12, contains: 2197


# number of pos scans = 12562
# number of scans in Y://2019_01 =  30108
# number of scans in Y://2019_02 =  26125
# number of scans in Y://2019_03 =  25709
# number of scans in Y://2019_04 =  26979
# number of scans in Y://2019_05 =  28088
# number of scans in Y://2019_06 =  25281
# number of scans in Z://2019_07 =  19850
# number of scans in Z://2019_08 =  25720

# number of scans in Y://2019_10 =  23312

# number of pos studies = 831
# number of studies in Z://2019_01 =  2567
# number of studies in Z://2019_02 =  2252
# number of studies in Z://2019_03 =  2186
# number of studies in Z://2019_04 =  2297
# number of studies in Z://2019_05 =  2397
# number of studies in Z://2019_06 =  2250
# number of studies in Z://2019_07 =  1746
# number of studies in Z://2019_08 =  2205
# number of studies in Z://2019_09 =  2392
# number of studies in Z://2019_10 =  2424
# number of studies in Z://2019_11 =  2472
# number of studies in Z://2019_12 =  2065
# studies_list = [2567, 2252, 2186, 2297, 2397, 2250, 1746, 2205,\
                #2392, 2424, 2472, 2065]
# number of healthy studies = 27253

# |pos ^ healthy| = 143

# approx 250MB/patient
# whole dataset: 6TB

# In[]
nii_size = np.array([2.87, 2.87, 24, 15, 2.3, 11, 2.5, 240, 17, 5.9, 46, 
                     500, 250, 260, 243, 1000, 2000, 100, 260, 1190, 200, 145])
scans_2019 = [30108, 26125, 25709, 26979, 28088, 25281, 19850, 25720] 
print(1000/60)