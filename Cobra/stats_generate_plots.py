#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 11:09:32 2021

@author: neus
"""

from statistics_functions import * 

# In[Plot single month]
main_path = '/home/neus/Documents/09.UCPH/MasterThesis/github/Thesis/Cobra/'
file_name = 'healthy_8'

tab_dir = main_path+'tables/'
fig_dir = main_path+'/figs/basic_stats'

#Read table
column_names,data = read_table(tab_dir+file_name+'.csv')

#Plots
plot_n_studies(column_names,data,fig_dir=fig_dir,file_name=file_name)
plot_n_scans(column_names,data,fig_dir=fig_dir,file_name=file_name)
plot_manufacturers(column_names,data,fig_dir=fig_dir,file_name=file_name)
plot_models(column_names,data,fig_dir=fig_dir,file_name=file_name)
plot_seq(column_names,data,fig_dir=fig_dir,file_name=file_name)

# In[Plot all 2019]
column_names,data = read_table(f'{tab_dir}healthy_1.csv')    
for i in range(2,9):
    column_names,current_data = read_table(f'{tab_dir}healthy_{i}.csv')    
    data = np.vstack((data,current_data))
    
#Plots
file_name = '2019_all'
tab_dir = main_path+'tables/'
fig_dir = main_path+'/figs/basic_stats'
main_path = '/home/neus/Documents/09.UCPH/MasterThesis/github/Thesis/Cobra/'

plot_n_studies(column_names,data,fig_dir=fig_dir,file_name=file_name)
plot_n_scans(column_names,data,fig_dir=fig_dir,file_name=file_name)
plot_manufacturers(column_names,data,fig_dir=fig_dir,file_name=file_name)
plot_models(column_names,data,fig_dir=fig_dir,file_name=file_name)
plot_seq(column_names,data,fig_dir=fig_dir,file_name=file_name)
