"""
Created on Mon Aug 30 10:32:43 2021
@author: klein
Counting number of patients, scans, and studies directly
"""


import access_sif_data.load_data_tools as ld
import os
from glob import iglob
from utilities import utils, basic
import json
import pydicom
from pathlib import Path
from utilities.basic import p


# In[main directories]
script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parent
base_data_dir = "Y:/"
out_pos_path = "Y:\\nii\\positive"
data_dirs = os.listdir(base_data_dir)
positive_dir = f"{base_data_dir}/positive"
healthy_dirs = sorted([f"{base_data_dir}/{x}" for x
                       in data_dirs if x.startswith('2019')])
print(base_dir)
print(f"main directories: {data_dirs}")

# In[Get all positive patients]
pos_patients_list = basic.list_subdir(positive_dir)

# In[Count patient numbers]
patient_numbers = []
for dir_ in healthy_dirs:
    patient_numbers.append(len(os.listdir(dir_)))
    print('.')
print(patient_numbers)

# In[Count study number]
study_counters = []
for dir_ in healthy_dirs:
    print(f"counting studies in {dir_}")
    study_counter = utils.count_subdirectories(dir_, 2)
    study_counters.append(study_counter)
    print(f'number of studies in {dir_} = {study_counter}')
print(study_counters)

# In[Count scans number]
scan_counters = {}
for healthy_dir in healthy_dirs[8:9]:
    print(f"counting studies in {healthy_dir}")
    patient_list = basic.list_subdir(healthy_dir)
    scan_counter = 0
    for pat_dir in patient_list:
        print('|', end=(''))
        scan_counter += len(ld.Patient(pat_dir).get_scan_directories())
    scan_counters[healthy_dir] = scan_counter
    print(f'number of scans in {healthy_dir} =  {scan_counter}')
# In[test]
print(base_dir)
# sum(1 for _ in iglob(f"{base_dir}/data/*/*"))
level3 = utils.count_subdirectories(f"{base_dir}/data", 2, True)
p(level3)
# In[Read some dcm files]

dir_ = "D:\\Thesis\\Cobra\\data\\0b630d10621e9c5d831a8053f95125b6\\5274cbd4b01b48a67071a35e252a692c\\MR\\5a4e9aa404ceb1f3995850b6e3ea246c"
subdir = basic.list_subdir(dir_)
dicom = pydicom.dcmread(subdir[0])
with open("D:/Thesis/Cobra/dicom.txt", "w") as f:
    f.write(str(dicom))

# In[]
out_patient_dir = os.path.join(
    "D:\Thesis\Cobra\data\0b630d10621e9c5d831a8053f95125b6_converted")
# In[look at converted patients]

con_pat_paths = [os.path.join(out_pos_path, conv_pat)
                 for conv_pat in conv_patients_list]
converted_patient = con_pat_paths[400]
files = basic.list_subdir(converted_patient)
print(files[:2])
# read json header
json_file = files[1]
with open(json_file) as f:
    header = json.load(f)
print(header)
#img_mat = nib.load(files[1])
#data = img_mat.get_fdata()
#print(f"the data shape is: {data.shape}")
#vis.display3d(data, axis=2, start_slice=10, num_slices=20)

# In[]
patient_list = basic.list_subdir(positive_dir)
# In[count studies]
study_counter = 0
for pat_dir in patient_list[:3]:
    print('|', end=(''))
    print(pat_dir)
    print([f for f in os.listdir(pat_dir)])
    study_counter += sum(1 for _ in iglob(pat_dir))
study_counters[positive_dir] = study_counter
print(f'number of studies in {positive_dir} =  {study_counter}')
# In[Count number of documented studies]
report_counters = {}
for dir_ in healthy_dirs[4:5]:
    patient_list = basic.list_subdir(dir_)
    report_counter = 0
    report_counter += sum(1 for _ in iglob(f"{dir_}/*/*/DOC/*/*.pdf"))
    print('|', end=(''))
    report_counters[dir_] = report_counter
    print(f'number of study reports in {dir_} = {report_counter}')


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
# 2392, 2424, 2472, 2065]
# number of healthy studies = 27253

# |pos ^ healthy| = 143

