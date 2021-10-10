# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 16:15:47 2021

@author: klein
"""

from utilss import call_mpunet
import os
from pathlib import Path
from utilss import utils
from data_access import load_data_tools as ld


def p(x): print(x)
# In[define paths]
script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parent
data_path = os.path.join(base_dir, 'data', 'data_folder')
 
# In[Reshape data]

test_img_path = f"{data_path}/test/images"
test_img_subdir = utils.list_subdir(test_img_path)
img_mat = ld.nii_data(test_img_subdir[0])
print(img_mat.shape)
