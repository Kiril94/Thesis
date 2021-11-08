#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 13:12:03 2021

@author: neus
"""

from data_access.load_data_tools import save_patientsIDlist
import pandas as pd
from utilss.utils import read_pos_data,read_whole_data

main_dir = '/home/neus/Documents/09.UCPH/MasterThesis/github/Thesis/Cobra'
tab_dir = main_dir+'/tables/'

healty_table = read_whole_data()
pos_table = read_pos_data()

# In[Find common patients]
healthy_patients = healty_table.drop_duplicates(subset='PatientID')['PatientID']
pos_patients = pos_table.drop_duplicates(subset='PatientID')['PatientID']

common_patients = list(set(healthy_patients) & set(pos_patients))

df_common = pd.DataFrame({'PatientID':common_patients})
df_common.to_csv(f'{tab_dir}ids_commonToBothPos2019.csv')

# In[Create table to send to computerome]

list_1 = np.array([1,2,3])
list_2 = np.array([4,5,6,7])

list_1+=list_2
