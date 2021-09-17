# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 10:58:38 2021

@author: klein
"""

import shutil
import os
from pathlib import Path
from utilss import utils


# In[tables directories]
org_base_dir = "Y:"
data_dirs = os.listdir(org_base_dir)
healthy_dirs = sorted([f"{org_base_dir}/{x}" for x \
                       in data_dirs if x.startswith('2019')])
target_base_dir = 
    out_pos_path = "Y:\\nii"

positive_dir = f"{base_data_dir}/positive" 

# In[move]

original = utils.list_subdir(table_dir)[0]
target = f"{base_dir}/test.csv"
shutil.move(original, target)