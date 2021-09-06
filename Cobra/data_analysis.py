"""
Created on Mon Aug 30 10:32:43 2021
@author: klein
"""

import data_access.load_data_tools as ld
import os
from glob import iglob
from vis import vis
import importlib
from pydicom import dcmread
from utilss import utils
import nibabel as nib
import time
from utilss import dicom2nifti
import datetime
importlib.reload(ld)
def p(string): print(string)

# In[main directories]
#base_data_dir = "/run/user/1000/gvfs/sftp:host=sif-io.erda.dk"
#data_dirs = os.listdir(base_data_dir)
#positive_dir = f"{base_data_dir}/positive" 
#healthy_dirs = [f"{base_data_dir}/{x}" for x in data_dirs if x.startswith('2019')]
#target_dir = f"{base_data_dir}/2019_01" 

base_data_dir = "Z:/"
out_pos_path = "Z:\\nii\\positive"
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
target_patients_list = utils.list_subdir(target_dir)
# In[Test]
if not False:
    print('a')
# In[Convert positive patients]
out_pos_path = "/run/user/1000/gvfs/sftp:host=sif-io.erda.dk/nii/2019_01"
start = time.time()
n_total_patients = len(target_patients_list)
i= 1352
for patient_dir in target_patients_list[i:]:
    m, s = divmod(time.time()-start , 60)
    h, m = divmod(m, 60)
    i+=1
    print(f'Patient {i:d} out of {n_total_patients:d} [{h:2.0f}h{m:2.0f}m{s:2.0f}s]')

# In[non converted patients]
pos_patients_id = [os.path.split(dir_)[1] for dir_ in pos_patients_list]
non_conv_patients = set(pos_patients_id)- set(conv_patients_list)
print(f"non converted patients:{len(non_conv_patients)}")
non_conv_patients_dirs = sorted([os.path.join(positive_dir, non_conv_pat)\
                          for non_conv_pat in non_conv_patients])
# In[Convert positive patients]

start = time.time()
patient_counter = len(non_conv_patients)
for patient_dir in non_conv_patients_dirs:
    patient_timer = time.time()
    patient = ld.Patient(patient_dir)
    patient_id = patient.get_id()
    scan_dirs = patient.get_scan_directories()
    out_patient_dir = os.path.join(out_pos_path, patient_id)
    if not os.path.exists(out_patient_dir):
        os.makedirs(out_patient_dir)
        print(f"{out_patient_dir} created")
    for scan_dir in scan_dirs:
        _, scan_id = os.path.split(scan_dir)
        out_path = os.path.join(out_patient_dir)
        dicom2nifti.dcm2nii(scan_dir, out_path)
        print('|',end=(''))
    patient_counter -= 1
    print(patient_counter)
    print(str(datetime.datetime.now()))
stop = time.time()
print(f"The conversion took: {stop-start} s")
# In[Look at one patient with subdirectories]
# test_pat = "Z:/positive/00e520dd9e4c7f2b7798263bd0916221/2d8ef0eb9e77c14475dad00723fb0ca7/MR/2c76b30765e19a46b140d0d07df70bb5/0e04a266d7b274469583b4044728b9a4.dcm"
# pos_patient_dir = pos_patients_list[20]
# p0 = ld.Patient(pos_patient_dir)
# p0_scandir = p0.get_scan_directories()
# subdirs = utils.list_subdir(p0_scandir[0])
# #for p in p0_scandir:
# #    print(dcmread(utils.list_subdir(p)[0]).SOPInstanceUID)
# #patient_ids = [p0.scan_dictionary(n, reconstruct_3d=False).PatientID \
# #              for n in range(len(p0_scandir))]
# print(dcmread(test_pat))
# #for sub in subdirs:
# #    print(dcmread(sub).SeriesInstanceUID)
# #print(p0.get_scan_dictionary(0))
# #print(p0_scandir[2])

# In[look at ni header]
# path_all = "0.nii"
# nii_path = os.path.join(out_path,path_all )
# img_mat = nib.load(nii_path)
# data = img_mat.get_fdata()
# print(img_mat.header)
# vis.display3d(data, axis=2)
# # In[Count Patients]
# healthy_count = 0
# for subdir in healthy_dirs:
#     healthy_count += count_subdirectories(subdir)
#     print(f"subdir: {subdir}, accumulated sum: {healthy_count}")
# print(healthy_count)

        

# In[Count scans number]
scan_counters = {}
for healthy_dir in healthy_dirs[6:]:
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
for healthy_dir in healthy_dirs:
    print(f"counting studies in {healthy_dir}")
    patient_list = utils.list_subdir(healthy_dir)
    study_counter = 0
    for pat_dir in patient_list:
        print('|',end=(''))
        study_counter += sum(1 for _ in iglob(pat_dir))
    study_counters[healthy_dir] = study_counter
    print(f'number of studies in {healthy_dir} =  {study_counter}')
# In[Test]
start_glob = time.time()
sum(1 for _ in iglob(healthy_dirs[0]))
print(time.time()-start_glob)
start_list = time.time()
len(os.listdir(healthy_dirs[0]))
print(time.time()-start_glob)
# In[Count number of documented studies]
report_counters = {}
for dir_ in healthy_dirs:
    patient_list = utils.list_subdir(dir_)
    report_counter = 0
    for pat_dir in patient_list:
        report_counter += sum(1 for _ in iglob(f"{pat_dir}/*/DOC"))
    report_counters[dir_] = report_counter 
    print(f'number of study reports in {dir_} =  {report_counter}')
# In[Look at one patient with subdirectories]
test_pat = "Z:/positive/00e520dd9e4c7f2b7798263bd0916221/\
    2d8ef0eb9e77c14475dad00723fb0ca7/MR/\
        2c76b30765e19a46b140d0d07df70bb5/\
            0e04a266d7b274469583b4044728b9a4.dcm"
#pos_patient_dir = pos_patients_list[20]
#p0 = ld.Patient(pos_patient_dir)
#p0_scandir = p0.get_scan_directories()
#subdirs = utils.list_subdir(p0_scandir[0])
#for p in p0_scandir:
#    print(dcmread(utils.list_subdir(p)[0]).SOPInstanceUID)
#patient_ids = [p0.scan_dictionary(n, reconstruct_3d=False).PatientID \
#              for n in range(len(p0_scandir))]
print(dcmread(test_pat))
#for sub in subdirs:
#    print(dcmread(sub).SeriesInstanceUID)
#print(p0.get_scan_dictionary(0))
#print(p0_scandir[2])

# In[look at ni header]
path_all = "0.nii"
nii_path = os.path.join(out_path,path_all )
img_mat = nib.load(nii_path)
data = img_mat.get_fdata()
print(img_mat.header)
vis.display3d(data, axis=2)
# In[Count Patients]
healthy_count = 0
for subdir in healthy_dirs:
    healthy_count += count_subdirectories(subdir)
    print(f"subdir: {subdir}, accumulated sum: {healthy_count}")
print(healthy_count)
# In[]

# number of pos patients = 831
# number of neg patients = 24908

# number of pos scans = 12562
# number of scans in Y://2019_01 =  30108
# number of scans in Y://2019_02 =  26125
# number of scans in Y://2019_03 =  25709
# number of scans in Y://2019_04 =  26979
# number of scans in Y://2019_05 =  28088
# number of scans in Y://2019_06 =  25281
# number of scans in Z://2019_07 =  19850
# number of scans in Z://2019_08 =  25720

# number of studies in Z://2019_01 =  2567
# number of studies in Z://2019_02 =  2252
# number of studies in Z://2019_03 =  2186
# number of studies in Z://2019_04 =  2297
# number of studies in Z://2019_05 =  2397
# number of studies in Z://2019_06 =  2250
# number of studies in Z://2019_07 =  1746

# approx 250MB/patient
# whole dataset: 6TB

# In[]
iterator = iglob("Z:/positive/00e520dd9e4c7f2b7798263bd0916221/*/DOC")
#iterator = os.listdir("Z:/positive/00e520dd9e4c7f2b7798263bd0916221")

print(sum(1 for _ in iterator))
#for i in iterator:
#    print(i)
