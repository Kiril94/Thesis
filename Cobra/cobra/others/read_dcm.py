#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 11:19:04 2021

@author: neus
"""
import matplotlib.pyplot as plt
from pydicom import dcmread
from pydicom.data import get_testdata_files
import numpy as np

import glob

#Example of SWI 
patient_id = '010cc5be2a398bc7e7cad3d9d78f3c26'
study_id = '59ab51a416dadda0d00267e463be027b'
series_id  = 'd8af76b86e06419be52edc9384553d3e'
sif_path = '/home/neus/sif'
# file_path = glob.glob(f'{sif_path}/*/{patient_id}/{study_id}/MR/{series_id}/') #to find the path


path_dcm=f'{sif_path}/*/{patient_id}/{study_id}/MR/{series_id}/'
files_name = glob.glob(path_dcm+'*.dcm')

for file in files_name:
    dcm_slice = dcmread(file)
    print(dcm_slice.SeriesDescription)


# dcm_slice0 = dcmread(files_name[0])
# print(type(dcm_slice0))
# print(dcm_slice0)


# #Read dcm files and sort them
# dcm_slices = [dcmread(file) for file in files_name]
# dcm_slices.sort(key=lambda x: int(x.InstanceNumber))

# #Plot them in a grid
# fig,ax = plt.subplots(5,5,figsize=(50,50))
# ax = ax.flatten()
# d3_volume = []
# for i in range(len(dcm_slices)):
#     ds = dcm_slices[i]
#     print(ds.PatientID)
#     ax[i].imshow(ds.pixel_array, cmap=plt.cm.gray)
#     d3_volume.append(ds.pixel_array)

# d3_volume = np.array(d3_volume)
