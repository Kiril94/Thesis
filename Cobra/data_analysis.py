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
#from utils import dicom2nifti
import dicom2nifti as d2n
import nibabel as nib
import glob
import pydicom
#from nipype.interfaces.dcm2nii import Dcm2niix
import sys
import json
import time
#importlib.reload(dicom2nifti)
#import dcm2niix
importlib.reload(ld)
def p(string): print(string)
import dcmstack


# In[]
import os
dcm2niix_dir = "D:\\Thesis\\Cobra\\helper\\dcm2niix_win\\dcm2niix.exe"
in_path = p0_scandir[0]#"Z:\\positive\\00e520dd9e4c7f2b7798263bd0916221\\2d8ef0eb9e77c14475dad00723fb0ca7\\MR\\2c76b30765e19a46b140d0d07df70bb5"
out_path = "D:\\Thesis\\Cobra\\data\\dicom2nifti\\p3"
#os.system(f"{dcm2niix_dir} -h")
start = time.time()
os.system(f"cmd /k {dcm2niix_dir} -o {out_path} {in_path}")
end = time.time()
print(end - start)
# In[Look at the result]
path = "2c76b30765e19a46b140d0d07df70bb5_MPR_Thick_Range[4]_0_104.json"
with open(os.path.join(out_path, path), 'r') as f:
	data = json.load(f)
print(data)
# In[Di2nii]
def convert_dcm_2_nii_x(dcm_folder, output_folder):
    converter = Dcm2niix()
    converter.inputs.source_dir = dcm_folder
    converter.inputs.output_dir = output_folder
    converter.inputs.compress = 'i'
    converter.run()

# In[main directories]

base_data_dir = "Z:/"
data_dirs = os.listdir(base_data_dir)
positive_dir = f"{base_data_dir}/positive" 
healthy_dirs = [f"{base_data_dir}/{x}" for x in data_dirs if x.startswith('2019')]
print(f"main directories: {data_dirs}")
# In[all positives]
pos_patients_list = utils.list_subdir(positive_dir)
# In[Look at one patient with subdirectories]
pos_patient_dir = pos_patients_list[20]
p0 = ld.Patient(pos_patient_dir)
p0_scandir = p0.get_scan_directories()
#for p in p0_scandir:
#    print(dcmread(utils.list_subdir(p)[0]).SOPInstanceUID)
#patient_ids = [p0.scan_dictionary(n, reconstruct_3d=False).PatientID \
#              for n in range(len(p0_scandir))]
print(p0_scandir[2])
# In[try dicom2nifti]
d2n.convert_directory(p0_scandir[0], "D:\\Thesis\\Cobra\\data\\dicom2nifti\\p1_python")
# In[try new script to convert]
convert_dcm_2_nii_x(p0_scandir[0], "D:\\Thesis\\Cobra\\data\\dicom2nifti\\p1_matlab")
# In[]
scan_dir = "Z:\\positive\\00e520dd9e4c7f2b7798263bd0916221\\2d8ef0eb9e77c14475dad00723fb0ca7\\MR\\2c76b30765e19a46b140d0d07df70bb5"
file_dir = utils.list_subdir(scan_dir)[0]
file = pydicom.read_file(file_dir)
print(file)

# In[Try dcmstack]
src_dcms = utils.list_subdir(p0_scandir[0])
stacks = dcmstack.parse_and_stack(src_dcms)
# In[]
#help(dcmstack.parse_and_stack)
print(dir(stacks))
#print(stacks.values())
stack = stacks.values()
nii = stack.to_nifti()
nii.to_filename("D:\\Thesis\\Cobra\\data\\dicom2nifti\\p1_py\\test.nii")
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
path_roi = "2c76b30765e19a46b140d0d07df70bb5_MPR_Thick_Range[4]_0_104a_ROI1.nii"
path_all = "2c76b30765e19a46b140d0d07df70bb5_MPR_Thick_Range[4]_0_104.nii"
nii_path = os.path.join(out_path,path_all )
img_mat = nib.load(nii_path)
data = img_mat.get_fdata()
print(data.shape)
vis.display3d(data, axis=1)
# In[]


# In[Count Patients]
healthy_count = 0
for subdir in healthy_dirs:
    healthy_count += count_subdirectories(subdir)
    print(f"subdir: {subdir}, accumulated sum: {healthy_count}")
print(healthy_count)
# In[]











