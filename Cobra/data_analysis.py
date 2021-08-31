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
importlib.reload(ld)


def p(string): print(string)
# In[main directories]

base_data_dir = "Z:"
data_dirs = os.listdir(base_data_dir)
positive_dir = f"{base_data_dir}/positive"
healthy_dirs = [f"Z:/{x}" for x in data_dirs if x.startswith('2019')]
print(healthy_dirs)
# In[Look at one patient with subdirectories]
pos_patient_dir = "Z:\\positive\\f430460b276e6618ad2c73daf269c228"
patient0 = ld.Patient(pos_patient_dir)
#arr = patient0.reconstruct3d(patient0.get_scan_directories()[0])
patient0.show_scan(1, {'axis':0})


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










