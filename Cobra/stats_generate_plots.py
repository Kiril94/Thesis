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

# In[]

df = pd.DataFrame(data,columns=column_names)
scan_months = np.array([int(date[5:7]) for date in df['InstanceCreationDate'].dropna()])
svis.nice_histogram(scan_months, np.arange(.5,13.5), show_plot=(True), 
                    xlabel='month', save=(True), 
                    figname=f"{fig_dir}/{file_name}/scan_months.png" )

unique_studies = df.drop_duplicates(subset='StudyInstanceUID')
studies_months = np.array([int(date[5:7]) for date in unique_studies['InstanceCreationDate'].dropna()])
svis.nice_histogram(scan_months, np.arange(.5,13.5), show_plot=(True), 
                    xlabel='month', save=(True), 
                    figname=f"{fig_dir}/{file_name}/studies_months.png" )
