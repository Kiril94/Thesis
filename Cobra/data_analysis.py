"""
Created on Mon Aug 30 10:32:43 2021
@author: klein
"""

import data_access.load_data_tools as ld
import os
#from vis import vis
import importlib
from pydicom import dcmread
from utilss import utils
import nibabel as nib
import glob
import pydicom
import sys
import json
import time
from utilss import dicom2nifti
importlib.reload(ld)
def p(string): print(string)

# In[main directories]
base_data_dir = "/run/user/1000/gvfs/sftp:host=sif-io.erda.dk"
data_dirs = os.listdir(base_data_dir)
positive_dir = f"{base_data_dir}/positive" 
healthy_dirs = [f"{base_data_dir}/{x}" for x in data_dirs if x.startswith('2019')]
target_dir = f"{base_data_dir}/2019_01" 
print(f"main directories: {data_dirs}")
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
    
    patient = ld.Patient(patient_dir)
    patient_id = patient.get_id()
    scan_dirs = patient.get_scan_directories()
    out_patient_dir = os.path.join(out_pos_path, patient_id)
    if not os.path.exists(out_patient_dir):
        os.makedirs(out_patient_dir)
    for scan_dir in scan_dirs[:1]:
        _, scan_id = os.path.split(scan_dir)
        out_path = out_patient_dir #os.path.join(out_patient_dir, scan_id)
        print(out_path)
        dicom2nifti.dcm2nii(scan_dir, out_path)
stop = time.time()
print(f"The conversion took: {stop-start} s")
# # In[Look at one patient with subdirectories]
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

# # In[look at ni header]
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