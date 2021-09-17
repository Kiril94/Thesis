#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 12:10:46 2021

@author: neus
"""

import timeit
import os 
import matplotlib.pyplot as plt
import numpy as np

# In[]

times = []
sizes = np.empty((9,10))
for i in range(1,10):
    
    
    times_same_comp = []
    for j in range(1,11):
    
        set_up = 'import os'
        code_to_test = f'os.system("dcm2niix -w 1 -z y -f CompressedMR_{j} -{i} -o /home/neus/Documents/09.UCPH/MasterThesis/ucph_sif_data/test_conversion/comp{i} /home/neus/Documents/09.UCPH/MasterThesis/ucph_sif_data/test_conversion/originals/MR_{j}")'
    
        times_same_comp.append(timeit.timeit(code_to_test,setup=set_up,number=10))
        sizes[i-1,j-1] = os.path.getsize(f'/home/neus/Documents/09.UCPH/MasterThesis/ucph_sif_data/test_conversion/comp{i}/CompressedMR_{j}.nii.gz')
    
    times.append(np.average(np.array(times_same_comp)))

sizes = np.average(sizes,axis=1)
    
    
    
# In[]

fig,ax = plt.subplots()
ax.plot(range(1,10),times,'-o')
ax.set(xlabel='Compression level',ylabel='Elapsed time (s)')
fig.show()

fig,ax = plt.subplots()
ax.plot(range(1,10),sizes,'-o')
ax.set(xlabel='Compression level',ylabel='Size (bytes)')
fig.show()