# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 10:32:43 2021

@author: klein
"""

import data_access.load_data_tools as ld
import os
from vis import display3d
import importlib
from pydicom import dcmread
from utils import utils
importlib.reload(ld)


def p(string): print(string)
# In[Specify Directories]

base_data_dir = "Z:"
data_dirs = os.listdir(base_data_dir)
#all_dirs = [x[0] for x in os.walk(base_data_dir)]
positive_dir = f"{base_data_dir}/positive"
healthy_dirs = [os.path.join(base_data_dir, x) for x in data_dirs if x.startswith('2019')]

#healthy_patients_dir_month01 = [os.path.join(healthy_dirs[0], x) for x in os.listdir(healthy_dirs[0])]
#example_healthy_patient_dir_month01 = "Z:/2019_01/00ccc73189d43bb9f4123d92ea3f19c1"
#patient00_scans_dirs = [os.path.join(patient00_dir, x) for x in os.listdir(patient00_dir)]
#print(patient00_scans_dirs)


# In[Look at one patient with subdirectories]
pos_patient_dir = "Z:\\positive\\f430460b276e6618ad2c73daf269c228"
patient0 = ld.Patient(pos_patient_dir)
#p(patient0.get_scan_directories())
#arr = patient0.reconstruct3d(patient0.get_scan_directories()[0])
patient0.show_scan(1, {'axis':0})




# In[]
pos_patient_directories = utils.list_subdir(positive_dir)
p(pos_patient_directories)
