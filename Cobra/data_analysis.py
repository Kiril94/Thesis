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
importlib.reload(utils)
importlib.reload(ld)
def p(string): print(string)
# In[main directories]

base_data_dir = "Z:/"
data_dirs = os.listdir(base_data_dir)
positive_dir = f"{base_data_dir}/positive"
healthy_dirs = [f"{base_data_dir}/{x}" for x in data_dirs if x.startswith('2019')]
print(data_dirs)
# In[Look at one patient with subdirectories]
pos_patient_dir = "Z:/positive/00e520dd9e4c7f2b7798263bd0916221"
p0 = ld.Patient(pos_patient_dir)
p0_scandir = p0.get_scan_directories()
patient_ids = [p0.scan_dictionary(n, reconstruct_3d=False).PatientID \
              for n in range(len(p0_scandir))]
# In[]
#p0_dicomdir = os.path.join(p0_scandir[0], os.listdir(p0_scandir[0])[0])
for sd, cd in zip(p0_scandir, patient_ids):
    print(f"scan dir: {sd[-5:]}, date:{cd}")
    #p(f"date: {cd.month} {cd.day}")
#print(p0_scandir)
#print(os.path.listdir(pos_patient_dir))
# In[convert di2nifti]
nifti_out_dir = "D:/Thesis/Cobra/data/dicom2nifti/p0"
#patient0_dicom = 
#arr = patient0.reconstruct3d(patient0.get_scan_directories()[0])
#patient0.show_scan(1, {'axis':0})

# In[]
def count_subdirectories(dir_):
    return sum(os.path.isdir(os.path.join(dir_,x)) for x \
               in os.listdir(dir_))

# In[]
healthy_count = 0
for subdir in healthy_dirs:
    healthy_count += count_subdirectories(subdir)
    print(healthy_count)
print(healthy_count)


# In[]











