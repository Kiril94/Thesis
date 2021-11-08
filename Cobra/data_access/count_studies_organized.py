#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 11:22:49 2021

@author: neus
"""

import sys
sys.path.append('../stats_tools/')

from statistics_functions import * 
import datetime
import pandas as pd

tables ='2019_all' #'positive'


main_dir = '/home/neus/Documents/09.UCPH/MasterThesis/github/Thesis/Cobra'
tab_dir = main_dir+'/tables/'
# In[Read data]

pos_months =  ['01_20','02_20','03_20','04_20','05_20','06_20','07_20','08_20','09_20','10_20','11_20','12_20','01_21','02_21','03_21','04_21','05_21','06_21','noDate']
neg_months =  ['01_18','02_18','03_18','04_18','05_18','06_18','07_18','08_18','09_18','10_18','11_18','12_18',
                '01_19','02_19','03_19','04_19','05_19','06_19','07_19','08_19','09_19','10_19','11_19','12_19',
                '01_20','02_20','03_20','04_20','05_20','06_20','07_20','08_20','09_20','10_20','11_20','12_20','01_21','02_21','03_21','04_21','05_21','06_21','noDate']


if (tables=='positive'):
    df = pd.read_csv(f'{tab_dir}pos_nn.csv',encoding= 'unicode_escape')
    df.to_csv(f'{tab_dir}pos_utf8.csv',encoding='utf-8')
    #In[Convert time and date to datetime for efficient access]
    time_k = 'InstanceCreationTime'
    date_k = 'InstanceCreationDate'
    df['DateTime'] = df[date_k] + ' ' +  df[time_k]
    #date_time_m = df['DateTime'].isnull()
    df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y%m%d %H:%M:%S')
    
    months_list = pos_months

elif (tables=='2019_all'):
    intersection_cols = ['PatientID',
      'ImageType',
      'AcquisitionDuration',
      'VariableFlipAngleFlag',
      'Manufacturer',
      'FrameOfReferenceUID',
      'AngioFlag',
      'InstanceCreationDate',
      'MagneticFieldStrength',
      'PixelPresentation',
      'SequenceVariant',
      'SliceThickness',
      'InversionTime',
      'PulseSequenceName',
      'AcquisitionContrast',
      'MRAcquisitionType',
      'NumberofPhaseEncodingSteps',
      'SpacingBetweenSlices',
      'SequenceName',
      'ManufacturerModelName',
      'ScanningSequence',
      'EchoTime',
      'EchoTrainLength',
      'NumberofAverages',
      'InstanceCreationTime',
      'ImagesInAcquisition',
      'PixelBandwith',
      'ImagingFrequency',
      'SecondEcho',
      'NumberOfEchoes',
      'EchoNumbers',
      'FlipAngle',
      'ImagedNuclues',
      'StudyInstanceUID',
      'PatientPosition',
      'SeriesInstanceUID',
      'ScanOptions',
      'SeriesDescription']
    
    df = pd.read_csv(f'{tab_dir}healthy_1.csv',encoding= 'unicode_escape')  
    df.to_csv(f'{tab_dir}healthy_1_utf8.csv',encoding='utf-8')
    data = df[intersection_cols].to_numpy()
    
    for i in range(2,10):
        df = pd.read_csv(f'{tab_dir}healthy_{i}.csv',encoding= 'unicode_escape')
        data = np.vstack((data,df[intersection_cols].to_numpy()))
        df.to_csv(f'{tab_dir}healthy_{i}_utf8.csv',encoding='utf-8')
       
    for name in ['healthy_10_n','healthy_11_nn','healthy_12_nn']:
        df = pd.read_csv(f'{tab_dir}{name}.csv',encoding= 'unicode_escape')
        data = np.vstack((data,df[intersection_cols].to_numpy()))
        df.to_csv(f'{tab_dir}healthy_{name[8:10]}_utf8.csv',encoding='utf-8')
    
    df = pd.DataFrame(data,columns=intersection_cols)
        
    #In[Convert time and date to datetime for efficient access]
    time_k = 'InstanceCreationTime'
    date_k = 'InstanceCreationDate'
    df['DateTime'] = df[date_k] + ' ' +  df[time_k]
    #date_time_m = df['DateTime'].isnull()
    df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y%m%d %H:%M:%S')
    
    months_list = neg_months
    
    df.to_csv(f'{tab_dir}healthy_all.csv',encoding='utf-8')
    
# In[Define time masks]

time_masks = {
    '01_18':(df['DateTime'] > datetime.datetime(2018,1,1)) & (df['DateTime'] < datetime.datetime(2018,1,31)),
    '02_18':(df['DateTime'] > datetime.datetime(2018,2,1)) & (df['DateTime'] < datetime.datetime(2018,2,28)),
    '03_18':(df['DateTime'] > datetime.datetime(2018,3,1)) & (df['DateTime'] < datetime.datetime(2018,3,31)),
    '04_18':(df['DateTime'] > datetime.datetime(2018,4,1)) & (df['DateTime'] < datetime.datetime(2018,4,30)),
    '05_18':(df['DateTime'] > datetime.datetime(2018,5,1)) & (df['DateTime'] < datetime.datetime(2018,5,31)),
    '06_18':(df['DateTime'] > datetime.datetime(2018,6,1)) & (df['DateTime'] < datetime.datetime(2018,6,30)),
    '07_18':(df['DateTime'] > datetime.datetime(2018,7,1)) & (df['DateTime'] < datetime.datetime(2018,7,31)),
    '08_18':(df['DateTime'] > datetime.datetime(2018,8,1)) & (df['DateTime'] < datetime.datetime(2018,8,31)),
    '09_18':(df['DateTime'] > datetime.datetime(2018,9,1)) & (df['DateTime'] < datetime.datetime(2018,9,30)),
    '10_18':(df['DateTime'] > datetime.datetime(2018,10,1)) & (df['DateTime'] < datetime.datetime(2018,10,31)),
    '11_18':(df['DateTime'] > datetime.datetime(2018,11,1)) & (df['DateTime'] < datetime.datetime(2018,11,30)),
    '12_18':(df['DateTime'] > datetime.datetime(2018,12,1)) & (df['DateTime'] < datetime.datetime(2018,12,31)),
    '01_19':(df['DateTime'] > datetime.datetime(2019,1,1)) & (df['DateTime'] < datetime.datetime(2019,1,31)),
    '02_19':(df['DateTime'] > datetime.datetime(2019,2,1)) & (df['DateTime'] < datetime.datetime(2019,2,28)),
    '03_19':(df['DateTime'] > datetime.datetime(2019,3,1)) & (df['DateTime'] < datetime.datetime(2019,3,31)),
    '04_19':(df['DateTime'] > datetime.datetime(2019,4,1)) & (df['DateTime'] < datetime.datetime(2019,4,30)),
    '05_19':(df['DateTime'] > datetime.datetime(2019,5,1)) & (df['DateTime'] < datetime.datetime(2019,5,31)),
    '06_19':(df['DateTime'] > datetime.datetime(2019,6,1)) & (df['DateTime'] < datetime.datetime(2019,6,30)),
    '07_19':(df['DateTime'] > datetime.datetime(2019,7,1)) & (df['DateTime'] < datetime.datetime(2019,7,31)),
    '08_19':(df['DateTime'] > datetime.datetime(2019,8,1)) & (df['DateTime'] < datetime.datetime(2019,8,31)),
    '09_19':(df['DateTime'] > datetime.datetime(2019,9,1)) & (df['DateTime'] < datetime.datetime(2019,9,30)),
    '10_19':(df['DateTime'] > datetime.datetime(2019,10,1)) & (df['DateTime'] < datetime.datetime(2019,10,31)),
    '11_19':(df['DateTime'] > datetime.datetime(2019,11,1)) & (df['DateTime'] < datetime.datetime(2019,11,30)),
    '12_19':(df['DateTime'] > datetime.datetime(2019,12,1)) & (df['DateTime'] < datetime.datetime(2019,12,31)),
    '01_20':(df['DateTime'] > datetime.datetime(2020,1,1)) & (df['DateTime'] < datetime.datetime(2020,1,31)),
    '02_20':(df['DateTime'] > datetime.datetime(2020,2,1)) & (df['DateTime'] < datetime.datetime(2020,2,28)),
    '03_20':(df['DateTime'] > datetime.datetime(2020,3,1)) & (df['DateTime'] < datetime.datetime(2020,3,31)),
    '04_20':(df['DateTime'] > datetime.datetime(2020,4,1)) & (df['DateTime'] < datetime.datetime(2020,4,30)),
    '05_20':(df['DateTime'] > datetime.datetime(2020,5,1)) & (df['DateTime'] < datetime.datetime(2020,5,31)),
    '06_20':(df['DateTime'] > datetime.datetime(2020,6,1)) & (df['DateTime'] < datetime.datetime(2020,6,30)),
    '07_20':(df['DateTime'] > datetime.datetime(2020,7,1)) & (df['DateTime'] < datetime.datetime(2020,7,31)),
    '08_20':(df['DateTime'] > datetime.datetime(2020,8,1)) & (df['DateTime'] < datetime.datetime(2020,8,31)),
    '09_20':(df['DateTime'] > datetime.datetime(2020,9,1)) & (df['DateTime'] < datetime.datetime(2020,9,30)),
    '10_20':(df['DateTime'] > datetime.datetime(2020,10,1)) & (df['DateTime'] < datetime.datetime(2020,10,31)),
    '11_20':(df['DateTime'] > datetime.datetime(2020,11,1)) & (df['DateTime'] < datetime.datetime(2020,11,30)),
    '12_20':(df['DateTime'] > datetime.datetime(2020,12,1)) & (df['DateTime'] < datetime.datetime(2020,12,31)),
    '01_21':(df['DateTime'] > datetime.datetime(2021,1,1)) & (df['DateTime'] < datetime.datetime(2021,1,31)),
    '02_21':(df['DateTime'] > datetime.datetime(2021,2,1)) & (df['DateTime'] < datetime.datetime(2021,2,28)),
    '03_21':(df['DateTime'] > datetime.datetime(2021,3,1)) & (df['DateTime'] < datetime.datetime(2021,3,31)),
    '04_21':(df['DateTime'] > datetime.datetime(2021,4,1)) & (df['DateTime'] < datetime.datetime(2021,4,30)),
    '05_21':(df['DateTime'] > datetime.datetime(2021,5,1)) & (df['DateTime'] < datetime.datetime(2021,5,31)),
    '06_21':(df['DateTime'] > datetime.datetime(2021,6,1)) & (df['DateTime'] < datetime.datetime(2021,6,30)),
    'noDate':(df['DateTime'].isnull()),
}

# In[Count scans over months]
n_volumes = {}
for month in months_list:
    n_volumes[month] = df[time_masks[month]].drop_duplicates(subset='SeriesInstanceUID').shape[0]


# In[Count studies over months]

n_studies = {}
for month in months_list[:-1]: #['01_20','02_20','03_20','04_20','05_20','06_20','07_20','08_20','09_20','10_20','11_20','12_20','01_21','02_21','03_21','04_21','05_21','06_21']:
    df_month = df[time_masks[month]].drop_duplicates(subset='StudyInstanceUID')
    df_p_sorted = df_month.groupby('PatientID').apply(
        lambda x: (x.sort_values(by=['DateTime'], ascending=True)))
    # Count the number of studies]
    patient_ids = df_p_sorted['PatientID'].unique()
    num_studies_l = []
    for patient in patient_ids:
        patient_mask = df_p_sorted['PatientID']==patient
        date_times = df_p_sorted[patient_mask]['DateTime'].values
        date_time0 = date_times[0] # [indexes[0]]
        study_counter = 1
        
        if (len(date_times)>1):
            for date_time in date_times[1:] :#[indexes[1]:]:
                try:
                    time_diff = date_time-date_time0
                    if time_diff.total_seconds()/3600>2:
                        study_counter += 1
                        date_time0 = date_time
                    else:
                        pass
                except:
                    pass
                    #print('NaT')
        num_studies_l.append(study_counter)
    n_studies[month] = sum(num_studies_l)

n_studies['noDate'] = df[time_masks['noDate']].drop_duplicates(subset='StudyInstanceUID').shape[0]


# In[Count patients over months]
n_patients = {}
for month in months_list:
    n_patients[month] = df[time_masks[month]].drop_duplicates(subset='PatientID').shape[0]

