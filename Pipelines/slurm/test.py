# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 16:15:50 2021

@author: klein
"""

import numpy as np

a = np.zeros(10)
b = np.ones(10)
c = a + b
np.savetxt('test.txt', c, fmt='%.18e')