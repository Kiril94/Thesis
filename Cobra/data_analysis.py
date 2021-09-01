# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 10:32:43 2021

@author: klein
"""

import data_access.load_data_tools as ld
import os
from vis import vis
import importlib
from pydicom import dcmread
from utils import utils
from utils import dicom2nifti
import nibabel as nib
from pydicom import dcmread
importlib.reload(dicom2nifti)
importlib.reload(ld)
def p(string): print(string)
# In[main directories]

base_data_dir = "Z:/"
data_dirs = os.listdir(base_data_dir)
positive_dir = f"{base_data_dir}/positive" 
healthy_dirs = [f"{base_data_dir}/{x}" for x in data_dirs if x.startswith('2019')]
print(data_dirs)
# In[all positives]
pos_patients_list = utils.list_subdir(positive_dir)
# In[Look at one patient with subdirectories]
pos_patient_dir = pos_patients_list[200]
p0 = ld.Patient(pos_patient_dir)
p0_scandir = p0.get_scan_directories()
#for p in p0_scandir:
#    print(dcmread(utils.list_subdir(p)[0]).SOPInstanceUID)
#patient_ids = [p0.scan_dictionary(n, reconstruct_3d=False).PatientID \
#              for n in range(len(p0_scandir))]
print(p0.info())
# In[]
#p0_dicomdir = os.path.join(p0_scandir[0], os.listdir(p0_scandir[0])[0])
for sd, cd in zip(p0_scandir, patient_ids):
    print(f"scan dir: {sd[-5:]}, date:{cd}")
    #p(f"date: {cd.month} {cd.day}")
#print(p0_scandir)
#print(os.path.listdir(pos_patient_dir))
# In[convert di2nifti]
dicom_dir = "Z:\\positive\\00e520dd9e4c7f2b7798263bd0916221\\2d8ef0eb9e77c14475dad00723fb0ca7\\MR\\2c76b30765e19a46b140d0d07df70bb5"
nifti_out_dir = "D:/Thesis/Cobra/data/dicom2nifti/p0"
dicom2nifti.dicom2nifti(dicom_dir, nifti_out_dir)
# In[look at ni header]

img = nib.load(nifti_out_dir+'.nii')
img_mat = nib.load("D:\\Thesis\\Cobra\\data\\dicom2nifti\\p0_matlab\\x_MPR_Thick_Range_4_.nii")

print(img.header)
print(img_mat.header)
# In[]
def count_subdirectories(dir_):
    return sum(os.path.isdir(os.path.join(dir_,x)) for x \
               in os.listdir(dir_))

# In[]
healthy_count = 0
for subdir in healthy_dirs:
    healthy_count += count_subdirectories(subdir)
    print(f"subdir: {subdir}, accumulated sum: {healthy_count}")
print(healthy_count)

# In[]
print(healthy_dirs)
# In[]
print(os.path.exists("Z://2019_08"))
# In[]











