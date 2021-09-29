# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 10:15:59 2021

@author: klein
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cryptography.fernet import Fernet

# In[Load and decrypt dataframe]
csv_folder = "D:/Thesis/Cobra/share/import"  
df = pd.read_csv(f"{csv_folder}/dst.csv")
key = b'1b3KCzziTwLPiqneoY8XMEQ2DhWpxixIeiRhLIWwZe4='
fernet = Fernet(key)
decrypt = lambda x: (fernet.decrypt(bytes(x[2:-1],'utf-8'))).decode('utf-8')
df.DST = df.DST.map(decrypt)
df.DST = df.DST.map(int)
# In[]

print((df.DST>0).sum())