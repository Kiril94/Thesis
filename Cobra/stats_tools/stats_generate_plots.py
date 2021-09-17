#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 11:09:32 2021

@author: neus
"""

from statistics_functions import * 
from utilss.utils import read_whole_data
import importlib

TE_k = 'EchoTime'
TR_k = 'RepetitionTime'
IR_k = 'InversionTime'
FA_k = 'FlipAngle'
SD_k = 'SeriesDescription'
PID_k = 'PatientID'
time_k = 'InstanceCreationTime'
date_k = 'InstanceCreationDate'
DT_k = 'DateTime'
SID_k = 'SeriesInstanceUID'

# In[Plot single month]
main_path = '/home/neus/Documents/09.UCPH/MasterThesis/github/Thesis/Cobra/'
file_name = 'healthy_8'

tab_dir = main_path+'tables/'
fig_dir = main_path+'/figs/basic_stats'

# #Read table
# column_names,data = read_table(tab_dir+file_name+'.csv')

# #Plots
# plot_n_studies(column_names,data,fig_dir=fig_dir,file_name=file_name)
# plot_n_scans(column_names,data,fig_dir=fig_dir,file_name=file_name)
# plot_manufacturers(column_names,data,fig_dir=fig_dir,file_name=file_name)
# plot_models(column_names,data,fig_dir=fig_dir,file_name=file_name)
# plot_seq(column_names,data,fig_dir=fig_dir,file_name=file_name)

# In[Plot all 2019]

    
#Plots
file_name = '2019_all'
tab_dir = main_path+'tables/'
fig_dir = main_path+'/figs/basic_stats'
main_path = '/home/neus/Documents/09.UCPH/MasterThesis/github/Thesis/Cobra/'

df = read_whole_data()
column_names = df.keys()
data = df.values

plot_n_studies(column_names,data,fig_dir=fig_dir,file_name=file_name)
plot_n_scans(column_names,data,fig_dir=fig_dir,file_name=file_name)
plot_manufacturers(column_names,data,fig_dir=fig_dir,file_name=file_name)
plot_models(column_names,data,fig_dir=fig_dir,file_name=file_name)
plot_seq(column_names,data,fig_dir=fig_dir,file_name=file_name)

# In[]
ps_datetime_count = df.groupby([df[DT_k].dt.year, df[DT_k].dt.month]).count()[SID_k]
year_month_keys = [str(int(key[1]))+'/'+str(key[0])[:4] for key in ps_datetime_count.keys()]
year_month_keys.insert(-1,'5/2021') # this month is missing
year_month_counts = ps_datetime_count.values
year_month_counts = np.insert(year_month_counts, -1, 0)
vis.bar_plot(year_month_keys[:-3], year_month_counts[:-3], figsize=(13,7),
             xtickparams_rot=70, 
                    xlabel='month/year', save_plot=(True), ylabel='Frequency',
                    title='Number of acquired volumes for 2019 patients',
                    figname=f"{fig_dir}/{file_name}/scans_months_years.png" )
# In[when is the date present but not a time]
p(f"{pd.isnull(df[date_k]).sum()} scans dont have a time or date")

# In[Study months distribution]
importlib.reload(stats)
_, study_dates = stats.time_between_studies(df)
# In[]
year_month_study_dates = [str(date.year)+'/'+str(date.month)for date in study_dates]
year_month_unique, year_month_counts = np.unique(
    np.array(year_month_study_dates), return_counts=True)
vis.bar_plot(year_month_unique[:-2], year_month_counts[:-2], figsize=(13,7),
             xtickparams_rot=70, 
                    xlabel='study month/year', save_plot=(True), ylabel='Frequency',
                    title='Studies for 2019 patients',
                    figname=f"{fig_dir}/{file_name}/studies_months_years.png" )

# In[]

time_diff_studies_pos, _ = stats.time_between_studies(df)

# In[]
svis.nice_histogram(np.array(time_diff_studies_pos)/24, 100, ylog_scale=(True),
                    show_plot=True, xlabel='Days between studies',
                    save=True, title='2019 Patients',
                    figname=f"{fig_dir}/{file_name}/time_between_studies.png")

