#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 17:04:35 2021

@author: neus
"""
import sys
sys.path.append('../stats_tools/')

from statistics_functions import * 
from functools import reduce
import datetime

# #Read data

# In[read all 2019]

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
data = df[intersection_cols].to_numpy()

for i in range(2,10):
    df = pd.read_csv(f'{tab_dir}healthy_{i}.csv',encoding= 'unicode_escape')
    data = np.vstack((data,df[intersection_cols].to_numpy()))
   
for name in ['healthy_10_n','healthy_11_nn','healthy_12_nn']:
    df = pd.read_csv(f'{tab_dir}{name}.csv',encoding= 'unicode_escape')
    data = np.vstack((data,df[intersection_cols].to_numpy()))

df = pd.DataFrame(data,columns=intersection_cols)
    
#In[Convert time and date to datetime for efficient access]
time_k = 'InstanceCreationTime'
date_k = 'InstanceCreationDate'
df['DateTime'] = df[date_k] + ' ' +  df[time_k]
#date_time_m = df['DateTime'].isnull()
df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y%m%d %H:%M:%S')

# In[Time masks definition]
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
    '02_21':(df['DateTime'] > datetime.datetime(2121,2,1)) & (df['DateTime'] < datetime.datetime(2121,2,28)),
    '03_21':(df['DateTime'] > datetime.datetime(2121,3,1)) & (df['DateTime'] < datetime.datetime(2121,3,31)),
    '04_21':(df['DateTime'] > datetime.datetime(2121,4,1)) & (df['DateTime'] < datetime.datetime(2121,4,30)),
    '05_21':(df['DateTime'] > datetime.datetime(2121,5,1)) & (df['DateTime'] < datetime.datetime(2121,5,31)),
    '06_21':(df['DateTime'] > datetime.datetime(2121,6,1)) & (df['DateTime'] < datetime.datetime(2121,6,30)),
    'noDate':(df['DateTime'].isnull()),
}

pos_months =  ['01_20','02_20','03_20','04_20','05_20','06_20','07_20','08_20','09_20','10_20','11_20','12_20','01_21','02_21','03_21','04_21','05_21','06_21','noDate']
neg_months =  ['01_18','02_18','03_18','04_18','05_18','06_18','07_18','08_18','09_18','10_18','11_18','12_18',
                '01_19','02_19','03_19','04_19','05_19','06_19','07_19','08_19','09_19','10_19','11_19','12_19',
                '01_20','02_20','03_20','04_20','05_20','06_20','07_20','08_20','09_20','10_20','11_20','12_20','01_21','02_21','03_21','04_21','05_21','06_21','noDate']




# # In[COUNTING STUDIES]
# n_studies = {}
# #In[2018]
# n_studies['01_18'] = df[(df['DateTime'] > datetime.datetime(2018,1,1)) & (df['DateTime'] < datetime.datetime(2018,1,31))].drop_duplicates(subset='StudyInstanceUID').shape[0]
# n_studies['02_18'] = df[(df['DateTime'] > datetime.datetime(2018,2,1)) & (df['DateTime'] < datetime.datetime(2018,2,28))].drop_duplicates(subset='StudyInstanceUID').shape[0]
# n_studies['03_18'] = df[(df['DateTime'] > datetime.datetime(2018,3,1)) & (df['DateTime'] < datetime.datetime(2018,3,31))].drop_duplicates(subset='StudyInstanceUID').shape[0]
# n_studies['04_18'] = df[(df['DateTime'] > datetime.datetime(2018,4,1)) & (df['DateTime'] < datetime.datetime(2018,4,30))].drop_duplicates(subset='StudyInstanceUID').shape[0]
# n_studies['05_18'] = df[(df['DateTime'] > datetime.datetime(2018,5,1)) & (df['DateTime'] < datetime.datetime(2018,5,31))].drop_duplicates(subset='StudyInstanceUID').shape[0]
# n_studies['06_18'] = df[(df['DateTime'] > datetime.datetime(2018,6,1)) & (df['DateTime'] < datetime.datetime(2018,6,30))].drop_duplicates(subset='StudyInstanceUID').shape[0]
# n_studies['07_18'] = df[(df['DateTime'] > datetime.datetime(2018,7,1)) & (df['DateTime'] < datetime.datetime(2018,7,31))].drop_duplicates(subset='StudyInstanceUID').shape[0]
# n_studies['08_18'] = df[(df['DateTime'] > datetime.datetime(2018,8,1)) & (df['DateTime'] < datetime.datetime(2018,8,31))].drop_duplicates(subset='StudyInstanceUID').shape[0]
# n_studies['09_18'] = df[(df['DateTime'] > datetime.datetime(2018,9,1)) & (df['DateTime'] < datetime.datetime(2018,9,30))].drop_duplicates(subset='StudyInstanceUID').shape[0]
# n_studies['10_18'] = df[(df['DateTime'] > datetime.datetime(2018,10,1)) & (df['DateTime'] < datetime.datetime(2018,10,31))].drop_duplicates(subset='StudyInstanceUID').shape[0]
# n_studies['11_18'] = df[(df['DateTime'] > datetime.datetime(2018,11,1)) & (df['DateTime'] < datetime.datetime(2018,11,30))].drop_duplicates(subset='StudyInstanceUID').shape[0]
# n_studies['12_18'] = df[(df['DateTime'] > datetime.datetime(2018,12,1)) & (df['DateTime'] < datetime.datetime(2018,12,31))].drop_duplicates(subset='StudyInstanceUID').shape[0]
# #In[2019]
# n_studies['01_19'] = df[(df['DateTime'] > datetime.datetime(2019,1,1)) & (df['DateTime'] < datetime.datetime(2019,1,31))].drop_duplicates(subset='StudyInstanceUID').shape[0]
# n_studies['02_19'] = df[(df['DateTime'] > datetime.datetime(2019,2,1)) & (df['DateTime'] < datetime.datetime(2019,2,28))].drop_duplicates(subset='StudyInstanceUID').shape[0]
# n_studies['03_19'] = df[(df['DateTime'] > datetime.datetime(2019,3,1)) & (df['DateTime'] < datetime.datetime(2019,3,31))].drop_duplicates(subset='StudyInstanceUID').shape[0]
# n_studies['04_19'] = df[(df['DateTime'] > datetime.datetime(2019,4,1)) & (df['DateTime'] < datetime.datetime(2019,4,30))].drop_duplicates(subset='StudyInstanceUID').shape[0]
# n_studies['05_19'] = df[(df['DateTime'] > datetime.datetime(2019,5,1)) & (df['DateTime'] < datetime.datetime(2019,5,31))].drop_duplicates(subset='StudyInstanceUID').shape[0]
# n_studies['06_19'] = df[(df['DateTime'] > datetime.datetime(2019,6,1)) & (df['DateTime'] < datetime.datetime(2019,6,30))].drop_duplicates(subset='StudyInstanceUID').shape[0]
# n_studies['07_19'] = df[(df['DateTime'] > datetime.datetime(2019,7,1)) & (df['DateTime'] < datetime.datetime(2019,7,31))].drop_duplicates(subset='StudyInstanceUID').shape[0]
# n_studies['08_19'] = df[(df['DateTime'] > datetime.datetime(2019,8,1)) & (df['DateTime'] < datetime.datetime(2019,8,31))].drop_duplicates(subset='StudyInstanceUID').shape[0]
# n_studies['09_19'] = df[(df['DateTime'] > datetime.datetime(2019,9,1)) & (df['DateTime'] < datetime.datetime(2019,9,30))].drop_duplicates(subset='StudyInstanceUID').shape[0]
# n_studies['10_19'] = df[(df['DateTime'] > datetime.datetime(2019,10,1)) & (df['DateTime'] < datetime.datetime(2019,10,31))].drop_duplicates(subset='StudyInstanceUID').shape[0]
# n_studies['11_19'] = df[(df['DateTime'] > datetime.datetime(2019,11,1)) & (df['DateTime'] < datetime.datetime(2019,11,30))].drop_duplicates(subset='StudyInstanceUID').shape[0]
# n_studies['12_19'] = df[(df['DateTime'] > datetime.datetime(2019,12,1)) & (df['DateTime'] < datetime.datetime(2019,12,31))].drop_duplicates(subset='StudyInstanceUID').shape[0]
# #In[2020]
# n_studies['01_20'] = df[(df['DateTime'] > datetime.datetime(2020,1,1)) & (df['DateTime'] < datetime.datetime(2020,1,31))].drop_duplicates(subset='StudyInstanceUID').shape[0]
# n_studies['02_20'] = df[(df['DateTime'] > datetime.datetime(2020,2,1)) & (df['DateTime'] < datetime.datetime(2020,2,28))].drop_duplicates(subset='StudyInstanceUID').shape[0]
# n_studies['03_20'] = df[(df['DateTime'] > datetime.datetime(2020,3,1)) & (df['DateTime'] < datetime.datetime(2020,3,31))].drop_duplicates(subset='StudyInstanceUID').shape[0]
# n_studies['04_20'] = df[(df['DateTime'] > datetime.datetime(2020,4,1)) & (df['DateTime'] < datetime.datetime(2020,4,30))].drop_duplicates(subset='StudyInstanceUID').shape[0]
# n_studies['05_20'] = df[(df['DateTime'] > datetime.datetime(2020,5,1)) & (df['DateTime'] < datetime.datetime(2020,5,31))].drop_duplicates(subset='StudyInstanceUID').shape[0]
# n_studies['06_20'] = df[(df['DateTime'] > datetime.datetime(2020,6,1)) & (df['DateTime'] < datetime.datetime(2020,6,30))].drop_duplicates(subset='StudyInstanceUID').shape[0]
# n_studies['07_20'] = df[(df['DateTime'] > datetime.datetime(2020,7,1)) & (df['DateTime'] < datetime.datetime(2020,7,31))].drop_duplicates(subset='StudyInstanceUID').shape[0]
# n_studies['08_20'] = df[(df['DateTime'] > datetime.datetime(2020,8,1)) & (df['DateTime'] < datetime.datetime(2020,8,31))].drop_duplicates(subset='StudyInstanceUID').shape[0]
# n_studies['09_20'] = df[(df['DateTime'] > datetime.datetime(2020,9,1)) & (df['DateTime'] < datetime.datetime(2020,9,30))].drop_duplicates(subset='StudyInstanceUID').shape[0]
# n_studies['10_20'] = df[(df['DateTime'] > datetime.datetime(2020,10,1)) & (df['DateTime'] < datetime.datetime(2020,10,31))].drop_duplicates(subset='StudyInstanceUID').shape[0]
# n_studies['11_20'] = df[(df['DateTime'] > datetime.datetime(2020,11,1)) & (df['DateTime'] < datetime.datetime(2020,11,30))].drop_duplicates(subset='StudyInstanceUID').shape[0]
# n_studies['12_20'] = df[(df['DateTime'] > datetime.datetime(2020,12,1)) & (df['DateTime'] < datetime.datetime(2020,12,31))].drop_duplicates(subset='StudyInstanceUID').shape[0]
# #In[2021]
# n_studies['01_21'] = df[(df['DateTime'] > datetime.datetime(2021,1,1)) & (df['DateTime'] < datetime.datetime(2021,1,31))].drop_duplicates(subset='StudyInstanceUID').shape[0]

# mask = df['DateTime'].isna()
# n_studies['noDate'] = df[mask].drop_duplicates(subset='StudyInstanceUID').shape[0]


# # {'01_18': 11,
# #  '02_18': 0,
# #  '03_18': 0,
# #  '04_18': 0,
# #  '05_18': 0,
# #  '06_18': 0,
# #  '07_18': 0,
# #  '08_18': 0,
# #  '09_18': 0,
# #  '10_18': 0,
# #  '11_18': 0,
# #  '12_18': 0,
# #  '01_19': 2395,
# #  '02_19': 2101,
# #  '03_19': 2103,
# #  '04_19': 2118,
# #  '05_19': 2276,
# #  '06_19': 2195,
# #  '07_19': 1639,
# #  '08_19': 2167,
# #  '09_19': 2238,
# #  '10_19': 2166,
# #  '11_19': 2403,
# #  '12_19': 2143,
# #  '01_20': 10,
# #  '02_20': 3,
# #  '03_20': 4,
# #  '04_20': 2,
# #  '05_20': 4,
# #  '06_20': 4,
# #  '07_20': 0,
# #  '08_20': 1,
# #  '09_20': 0,
# #  '10_20': 1,
# #  '11_20': 4,
# #  '12_20': 0,
# #  '01_21': 1,
# #  'noDate': 11499}


# # # In[Counting over folders]

# # n_studies = {}
# # for i in range(1,10):
# #     df = pd.read_csv(f'{tab_dir}healthy_{i}.csv',encoding= 'unicode_escape')
# #     num_studies = df.drop_duplicates(subset='StudyInstanceUID').shape[0]
# #     n_studies[f"healthy_{i}"] = num_studies
    
    
# # for name in ['healthy_10_n','healthy_11_nn','healthy_12_nn']:
# #     df = pd.read_csv(f'{tab_dir}{name}.csv',encoding= 'unicode_escape')
# #     num_studies = df.drop_duplicates(subset='StudyInstanceUID').shape[0]
# #     n_studies[name] = num_studies


# # # {'healthy_1': 2641,
# # #  'healthy_2': 2312,
# # #  'healthy_3': 2251,
# # #  'healthy_4': 2370,
# # #  'healthy_5': 2465,
# # #  'healthy_6': 2306,
# # #  'healthy_7': 1793,
# # #  'healthy_8': 2256,
# # #  'healthy_9': 2441,
# # #  'healthy_10_n': 2430,
# # #  'healthy_11_nn': 2539,
# # #  'healthy_12_nn': 2267}

# # # In[Counting over folders]

# # n_studies = {}
# # for i in range(1,10):
# #     df = pd.read_csv(f'{tab_dir}healthy_{i}.csv',encoding= 'unicode_escape')
# #     df = df.drop_duplicates(subset='StudyInstanceUID')
    
# #     time_k = 'InstanceCreationTime'
# #     date_k = 'InstanceCreationDate'
# #     df['DateTime'] = df[date_k] + ' ' +  df[time_k]
# #     df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y%m%d %H:%M:%S')

# #     # Sort the the scans by time and count those that are less than 2 hours apart]
# #     df_p_sorted = df.groupby('PatientID').apply(
# #         lambda x: (x.sort_values(by=['DateTime'], ascending=True)))
# #     # Count the number of studies]
# #     patient_ids = df_p_sorted['PatientID'].unique()
# #     num_studies_l = []
# #     for patient in patient_ids:
# #         patient_mask = df_p_sorted['PatientID']==patient
# #         date_times = df_p_sorted[patient_mask]['DateTime']
# #         date_time0 = date_times[0]
# #         study_counter = 1
# #         for date_time in date_times[1:]:
# #             try:
# #                 time_diff = date_time-date_time0
# #                 if time_diff.total_seconds()/3600>2:
# #                     study_counter += 1
# #                     date_time0 = date_time
# #                 else:
# #                     pass
# #             except:
# #                 print('NaT')
# #         num_studies_l.append(study_counter)
# #     n_studies[f"healthy_{i}"] = sum(num_studies_l)
# #     print('end')
 
# #     # In[]
# # for name in ['healthy_10_n','healthy_11_nn','healthy_12_nn']:
# #     df = pd.read_csv(f'{tab_dir}{name}.csv',encoding= 'unicode_escape')
# #     df = df.drop_duplicates(subset='StudyInstanceUID')
    
# #     time_k = 'InstanceCreationTime'
# #     date_k = 'InstanceCreationDate'
# #     df['DateTime'] = df[date_k] + ' ' +  df[time_k]
# #     df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y%m%d %H:%M:%S')

# #     # Sort the the scans by time and count those that are less than 2 hours apart]
# #     df_p_sorted = df.groupby('PatientID').apply(
# #         lambda x: (x.sort_values(by=['DateTime'], ascending=True)))
# #     # Count the number of studies]
# #     patient_ids = df_p_sorted['PatientID'].unique()
# #     num_studies_l = []
# #     for patient in patient_ids:
# #         patient_mask = df_p_sorted['PatientID']==patient
# #         date_times = df_p_sorted[patient_mask]['DateTime']
# #         date_time0 = date_times[0]
# #         study_counter = 1
# #         for date_time in date_times[1:]:
# #             try:
# #                 time_diff = date_time-date_time0
# #                 if time_diff.total_seconds()/3600>2:
# #                     study_counter += 1
# #                     date_time0 = date_time
# #                 else:
# #                     pass
# #             except:
# #                 print('NaT')
# #         num_studies_l.append(study_counter)
# #     n_studies[f"{name}"] = sum(num_studies_l)
# #     print('end')


# # {'healthy_1': 2617,
# #  'healthy_2': 2296,
# #  'healthy_3': 2228,
# #  'healthy_4': 2344,
# #  'healthy_5': 2438,
# #  'healthy_6': 2285,
# #  'healthy_7': 1780,
# #  'healthy_8': 2241,
# #  'healthy_9': 2425,
# #  'healthy_10_n': 2410,
# #  'healthy_11_nn': 2520,
# #  'healthy_12_nn': 2249}

# # In[studies negative patients per month]

# # In[Counting scans]

# n_scans = {}
# #In[2018]
# n_scans['01_18'] = df[(df['DateTime'] > datetime.datetime(2018,1,1)) & (df['DateTime'] < datetime.datetime(2018,1,31))].drop_duplicates(subset='SeriesInstanceUID').shape[0]
# n_scans['02_18'] = df[(df['DateTime'] > datetime.datetime(2018,2,1)) & (df['DateTime'] < datetime.datetime(2018,2,28))].drop_duplicates(subset='SeriesInstanceUID').shape[0]
# n_scans['03_18'] = df[(df['DateTime'] > datetime.datetime(2018,3,1)) & (df['DateTime'] < datetime.datetime(2018,3,31))].drop_duplicates(subset='SeriesInstanceUID').shape[0]
# n_scans['04_18'] = df[(df['DateTime'] > datetime.datetime(2018,4,1)) & (df['DateTime'] < datetime.datetime(2018,4,30))].drop_duplicates(subset='SeriesInstanceUID').shape[0]
# n_scans['05_18'] = df[(df['DateTime'] > datetime.datetime(2018,5,1)) & (df['DateTime'] < datetime.datetime(2018,5,31))].drop_duplicates(subset='SeriesInstanceUID').shape[0]
# n_scans['06_18'] = df[(df['DateTime'] > datetime.datetime(2018,6,1)) & (df['DateTime'] < datetime.datetime(2018,6,30))].drop_duplicates(subset='SeriesInstanceUID').shape[0]
# n_scans['07_18'] = df[(df['DateTime'] > datetime.datetime(2018,7,1)) & (df['DateTime'] < datetime.datetime(2018,7,31))].drop_duplicates(subset='SeriesInstanceUID').shape[0]
# n_scans['08_18'] = df[(df['DateTime'] > datetime.datetime(2018,8,1)) & (df['DateTime'] < datetime.datetime(2018,8,31))].drop_duplicates(subset='SeriesInstanceUID').shape[0]
# n_scans['09_18'] = df[(df['DateTime'] > datetime.datetime(2018,9,1)) & (df['DateTime'] < datetime.datetime(2018,9,30))].drop_duplicates(subset='SeriesInstanceUID').shape[0]
# n_scans['10_18'] = df[(df['DateTime'] > datetime.datetime(2018,10,1)) & (df['DateTime'] < datetime.datetime(2018,10,31))].drop_duplicates(subset='SeriesInstanceUID').shape[0]
# n_scans['11_18'] = df[(df['DateTime'] > datetime.datetime(2018,11,1)) & (df['DateTime'] < datetime.datetime(2018,11,30))].drop_duplicates(subset='SeriesInstanceUID').shape[0]
# n_scans['12_18'] = df[(df['DateTime'] > datetime.datetime(2018,12,1)) & (df['DateTime'] < datetime.datetime(2018,12,31))].drop_duplicates(subset='SeriesInstanceUID').shape[0]
# #In[2019]
# n_scans['01_19'] = df[(df['DateTime'] > datetime.datetime(2019,1,1)) & (df['DateTime'] < datetime.datetime(2019,1,31))].drop_duplicates(subset='SeriesInstanceUID').shape[0]
# n_scans['02_19'] = df[(df['DateTime'] > datetime.datetime(2019,2,1)) & (df['DateTime'] < datetime.datetime(2019,2,28))].drop_duplicates(subset='SeriesInstanceUID').shape[0]
# n_scans['03_19'] = df[(df['DateTime'] > datetime.datetime(2019,3,1)) & (df['DateTime'] < datetime.datetime(2019,3,31))].drop_duplicates(subset='SeriesInstanceUID').shape[0]
# n_scans['04_19'] = df[(df['DateTime'] > datetime.datetime(2019,4,1)) & (df['DateTime'] < datetime.datetime(2019,4,30))].drop_duplicates(subset='SeriesInstanceUID').shape[0]
# n_scans['05_19'] = df[(df['DateTime'] > datetime.datetime(2019,5,1)) & (df['DateTime'] < datetime.datetime(2019,5,31))].drop_duplicates(subset='SeriesInstanceUID').shape[0]
# n_scans['06_19'] = df[(df['DateTime'] > datetime.datetime(2019,6,1)) & (df['DateTime'] < datetime.datetime(2019,6,30))].drop_duplicates(subset='SeriesInstanceUID').shape[0]
# n_scans['07_19'] = df[(df['DateTime'] > datetime.datetime(2019,7,1)) & (df['DateTime'] < datetime.datetime(2019,7,31))].drop_duplicates(subset='SeriesInstanceUID').shape[0]
# n_scans['08_19'] = df[(df['DateTime'] > datetime.datetime(2019,8,1)) & (df['DateTime'] < datetime.datetime(2019,8,31))].drop_duplicates(subset='SeriesInstanceUID').shape[0]
# n_scans['09_19'] = df[(df['DateTime'] > datetime.datetime(2019,9,1)) & (df['DateTime'] < datetime.datetime(2019,9,30))].drop_duplicates(subset='SeriesInstanceUID').shape[0]
# n_scans['10_19'] = df[(df['DateTime'] > datetime.datetime(2019,10,1)) & (df['DateTime'] < datetime.datetime(2019,10,31))].drop_duplicates(subset='SeriesInstanceUID').shape[0]
# n_scans['11_19'] = df[(df['DateTime'] > datetime.datetime(2019,11,1)) & (df['DateTime'] < datetime.datetime(2019,11,30))].drop_duplicates(subset='SeriesInstanceUID').shape[0]
# n_scans['12_19'] = df[(df['DateTime'] > datetime.datetime(2019,12,1)) & (df['DateTime'] < datetime.datetime(2019,12,31))].drop_duplicates(subset='SeriesInstanceUID').shape[0]
# #In[2020]
# n_scans['01_20'] = df[(df['DateTime'] > datetime.datetime(2020,1,1)) & (df['DateTime'] < datetime.datetime(2020,1,31))].drop_duplicates(subset='SeriesInstanceUID').shape[0]
# n_scans['02_20'] = df[(df['DateTime'] > datetime.datetime(2020,2,1)) & (df['DateTime'] < datetime.datetime(2020,2,28))].drop_duplicates(subset='SeriesInstanceUID').shape[0]
# n_scans['03_20'] = df[(df['DateTime'] > datetime.datetime(2020,3,1)) & (df['DateTime'] < datetime.datetime(2020,3,31))].drop_duplicates(subset='SeriesInstanceUID').shape[0]
# n_scans['04_20'] = df[(df['DateTime'] > datetime.datetime(2020,4,1)) & (df['DateTime'] < datetime.datetime(2020,4,30))].drop_duplicates(subset='SeriesInstanceUID').shape[0]
# n_scans['05_20'] = df[(df['DateTime'] > datetime.datetime(2020,5,1)) & (df['DateTime'] < datetime.datetime(2020,5,31))].drop_duplicates(subset='SeriesInstanceUID').shape[0]
# n_scans['06_20'] = df[(df['DateTime'] > datetime.datetime(2020,6,1)) & (df['DateTime'] < datetime.datetime(2020,6,30))].drop_duplicates(subset='SeriesInstanceUID').shape[0]
# n_scans['07_20'] = df[(df['DateTime'] > datetime.datetime(2020,7,1)) & (df['DateTime'] < datetime.datetime(2020,7,31))].drop_duplicates(subset='SeriesInstanceUID').shape[0]
# n_scans['08_20'] = df[(df['DateTime'] > datetime.datetime(2020,8,1)) & (df['DateTime'] < datetime.datetime(2020,8,31))].drop_duplicates(subset='SeriesInstanceUID').shape[0]
# n_scans['09_20'] = df[(df['DateTime'] > datetime.datetime(2020,9,1)) & (df['DateTime'] < datetime.datetime(2020,9,30))].drop_duplicates(subset='SeriesInstanceUID').shape[0]
# n_scans['10_20'] = df[(df['DateTime'] > datetime.datetime(2020,10,1)) & (df['DateTime'] < datetime.datetime(2020,10,31))].drop_duplicates(subset='SeriesInstanceUID').shape[0]
# n_scans['11_20'] = df[(df['DateTime'] > datetime.datetime(2020,11,1)) & (df['DateTime'] < datetime.datetime(2020,11,30))].drop_duplicates(subset='SeriesInstanceUID').shape[0]
# n_scans['12_20'] = df[(df['DateTime'] > datetime.datetime(2020,12,1)) & (df['DateTime'] < datetime.datetime(2020,12,31))].drop_duplicates(subset='SeriesInstanceUID').shape[0]
# #In[2021]
# n_scans['01_21'] = df[(df['DateTime'] > datetime.datetime(2021,1,1)) & (df['DateTime'] < datetime.datetime(2021,1,31))].drop_duplicates(subset='SeriesInstanceUID').shape[0]

# mask = df['DateTime'].isna()
# n_scans['noDate'] = df[mask].drop_duplicates(subset='SeriesInstanceUID').shape[0]

# # {'01_18': 101,
# #  '02_18': 0,
# #  '03_18': 0,
# #  '04_18': 0,
# #  '05_18': 0,
# #  '06_18': 0,
# #  '07_18': 0,
# #  '08_18': 0,
# #  '09_18': 0,
# #  '10_18': 0,
# #  '11_18': 0,
# #  '12_18': 0,
# #  '01_19': 24389,
# #  '02_19': 21123,
# #  '03_19': 21842,
# #  '04_19': 21416,
# #  '05_19': 23064,
# #  '06_19': 20976,
# #  '07_19': 15820,
# #  '08_19': 21675,
# #  '09_19': 22461,
# #  '10_19': 17663,
# #  '11_19': 23938,
# #  '12_19': 21416,
# #  '01_20': 14,
# #  '02_20': 23,
# #  '03_20': 7,
# #  '04_20': 4,
# #  '05_20': 7,
# #  '06_20': 4,
# #  '07_20': 0,
# #  '08_20': 1,
# #  '09_20': 0,
# #  '10_20': 1,
# #  '11_20': 6,
# #  '12_20': 0,
# #  '01_21': 1,
# #  'noDate': 56607}

# # In[Counting scans with duplicates]

# n_scans = {}
# #In[2018]
# n_scans['01_18'] = df[(df['DateTime'] > datetime.datetime(2018,1,1)) & (df['DateTime'] < datetime.datetime(2018,1,31))].shape[0]
# n_scans['02_18'] = df[(df['DateTime'] > datetime.datetime(2018,2,1)) & (df['DateTime'] < datetime.datetime(2018,2,28))].shape[0]
# n_scans['03_18'] = df[(df['DateTime'] > datetime.datetime(2018,3,1)) & (df['DateTime'] < datetime.datetime(2018,3,31))].shape[0]
# n_scans['04_18'] = df[(df['DateTime'] > datetime.datetime(2018,4,1)) & (df['DateTime'] < datetime.datetime(2018,4,30))].shape[0]
# n_scans['05_18'] = df[(df['DateTime'] > datetime.datetime(2018,5,1)) & (df['DateTime'] < datetime.datetime(2018,5,31))].shape[0]
# n_scans['06_18'] = df[(df['DateTime'] > datetime.datetime(2018,6,1)) & (df['DateTime'] < datetime.datetime(2018,6,30))].shape[0]
# n_scans['07_18'] = df[(df['DateTime'] > datetime.datetime(2018,7,1)) & (df['DateTime'] < datetime.datetime(2018,7,31))].shape[0]
# n_scans['08_18'] = df[(df['DateTime'] > datetime.datetime(2018,8,1)) & (df['DateTime'] < datetime.datetime(2018,8,31))].shape[0]
# n_scans['09_18'] = df[(df['DateTime'] > datetime.datetime(2018,9,1)) & (df['DateTime'] < datetime.datetime(2018,9,30))].shape[0]
# n_scans['10_18'] = df[(df['DateTime'] > datetime.datetime(2018,10,1)) & (df['DateTime'] < datetime.datetime(2018,10,31))].shape[0]
# n_scans['11_18'] = df[(df['DateTime'] > datetime.datetime(2018,11,1)) & (df['DateTime'] < datetime.datetime(2018,11,30))].shape[0]
# n_scans['12_18'] = df[(df['DateTime'] > datetime.datetime(2018,12,1)) & (df['DateTime'] < datetime.datetime(2018,12,31))].shape[0]
# #In[2019]
# n_scans['01_19'] = df[(df['DateTime'] > datetime.datetime(2019,1,1)) & (df['DateTime'] < datetime.datetime(2019,1,31))].shape[0]
# n_scans['02_19'] = df[(df['DateTime'] > datetime.datetime(2019,2,1)) & (df['DateTime'] < datetime.datetime(2019,2,28))].shape[0]
# n_scans['03_19'] = df[(df['DateTime'] > datetime.datetime(2019,3,1)) & (df['DateTime'] < datetime.datetime(2019,3,31))].shape[0]
# n_scans['04_19'] = df[(df['DateTime'] > datetime.datetime(2019,4,1)) & (df['DateTime'] < datetime.datetime(2019,4,30))].shape[0]
# n_scans['05_19'] = df[(df['DateTime'] > datetime.datetime(2019,5,1)) & (df['DateTime'] < datetime.datetime(2019,5,31))].shape[0]
# n_scans['06_19'] = df[(df['DateTime'] > datetime.datetime(2019,6,1)) & (df['DateTime'] < datetime.datetime(2019,6,30))].shape[0]
# n_scans['07_19'] = df[(df['DateTime'] > datetime.datetime(2019,7,1)) & (df['DateTime'] < datetime.datetime(2019,7,31))].shape[0]
# n_scans['08_19'] = df[(df['DateTime'] > datetime.datetime(2019,8,1)) & (df['DateTime'] < datetime.datetime(2019,8,31))].shape[0]
# n_scans['09_19'] = df[(df['DateTime'] > datetime.datetime(2019,9,1)) & (df['DateTime'] < datetime.datetime(2019,9,30))].shape[0]
# n_scans['10_19'] = df[(df['DateTime'] > datetime.datetime(2019,10,1)) & (df['DateTime'] < datetime.datetime(2019,10,31))].shape[0]
# n_scans['11_19'] = df[(df['DateTime'] > datetime.datetime(2019,11,1)) & (df['DateTime'] < datetime.datetime(2019,11,30))].shape[0]
# n_scans['12_19'] = df[(df['DateTime'] > datetime.datetime(2019,12,1)) & (df['DateTime'] < datetime.datetime(2019,12,31))].shape[0]
# #In[2020]
# n_scans['01_20'] = df[(df['DateTime'] > datetime.datetime(2020,1,1)) & (df['DateTime'] < datetime.datetime(2020,1,31))].shape[0]
# n_scans['02_20'] = df[(df['DateTime'] > datetime.datetime(2020,2,1)) & (df['DateTime'] < datetime.datetime(2020,2,28))].shape[0]
# n_scans['03_20'] = df[(df['DateTime'] > datetime.datetime(2020,3,1)) & (df['DateTime'] < datetime.datetime(2020,3,31))].shape[0]
# n_scans['04_20'] = df[(df['DateTime'] > datetime.datetime(2020,4,1)) & (df['DateTime'] < datetime.datetime(2020,4,30))].shape[0]
# n_scans['05_20'] = df[(df['DateTime'] > datetime.datetime(2020,5,1)) & (df['DateTime'] < datetime.datetime(2020,5,31))].shape[0]
# n_scans['06_20'] = df[(df['DateTime'] > datetime.datetime(2020,6,1)) & (df['DateTime'] < datetime.datetime(2020,6,30))].shape[0]
# n_scans['07_20'] = df[(df['DateTime'] > datetime.datetime(2020,7,1)) & (df['DateTime'] < datetime.datetime(2020,7,31))].shape[0]
# n_scans['08_20'] = df[(df['DateTime'] > datetime.datetime(2020,8,1)) & (df['DateTime'] < datetime.datetime(2020,8,31))].shape[0]
# n_scans['09_20'] = df[(df['DateTime'] > datetime.datetime(2020,9,1)) & (df['DateTime'] < datetime.datetime(2020,9,30))].shape[0]
# n_scans['10_20'] = df[(df['DateTime'] > datetime.datetime(2020,10,1)) & (df['DateTime'] < datetime.datetime(2020,10,31))].shape[0]
# n_scans['11_20'] = df[(df['DateTime'] > datetime.datetime(2020,11,1)) & (df['DateTime'] < datetime.datetime(2020,11,30))].shape[0]
# n_scans['12_20'] = df[(df['DateTime'] > datetime.datetime(2020,12,1)) & (df['DateTime'] < datetime.datetime(2020,12,31))].shape[0]
# #In[2021]
# n_scans['01_21'] = df[(df['DateTime'] > datetime.datetime(2021,1,1)) & (df['DateTime'] < datetime.datetime(2021,1,31))].shape[0]

# mask = df['DateTime'].isna()
# n_scans['noDate'] = df[mask].shape[0]

# # {'01_18': 101,
# #  '02_18': 0,
# #  '03_18': 0,
# #  '04_18': 0,
# #  '05_18': 0,
# #  '06_18': 0,
# #  '07_18': 0,
# #  '08_18': 0,
# #  '09_18': 0,
# #  '10_18': 0,
# #  '11_18': 0,
# #  '12_18': 0,
# #  '01_19': 24389,
# #  '02_19': 21123,
# #  '03_19': 21842,
# #  '04_19': 21416,
# #  '05_19': 23064,
# #  '06_19': 20976,
# #  '07_19': 15820,
# #  '08_19': 21675,
# #  '09_19': 22461,
# #  '10_19': 17663,
# #  '11_19': 23938,
# #  '12_19': 21416,
# #  '01_20': 14,
# #  '02_20': 23,
# #  '03_20': 7,
# #  '04_20': 4,
# #  '05_20': 7,
# #  '06_20': 4,
# #  '07_20': 0,
# #  '08_20': 1,
# #  '09_20': 0,
# #  '10_20': 1,
# #  '11_20': 6,
# #  '12_20': 0,
# #  '01_21': 1,
# #  'noDate': 56607}

# # In[n_patients]

# n_scans_per_folder = {}
# for i in range(1,10):
#     df = pd.read_csv(f'{tab_dir}healthy_{i}.csv',encoding= 'unicode_escape')
#     n_scans_per_folder[f"healthy_{i}"] = df.drop_duplicates(subset='SeriesInstanceUID').shape[0]
    
    
# for name in ['healthy_10_n','healthy_11_nn','healthy_12_nn','pos_n']:
#     df = pd.read_csv(f'{tab_dir}{name}.csv',encoding= 'unicode_escape')
#     n_scans_per_folder[name] = df.drop_duplicates(subset='SeriesInstanceUID').shape[0]


# # # {'healthy_1': 2641,
# # #  'healthy_2': 2312,
# # #  'healthy_3': 2251,
# # #  'healthy_4': 2370,
# # #  'healthy_5': 2465,
# # #  'healthy_6': 2306,
# # #  'healthy_7': 1793,
# # #  'healthy_8': 2256,
# # #  'healthy_9': 2441,
# # #  'healthy_10_n': 2430,
# # #  'healthy_11_nn': 2539,
# # #  'healthy_12_nn': 2267}


# # In[COUNTING Patients]

# n_patients = {}
# #In[2018]
# n_patients['01_18'] = df[(df['DateTime'] > datetime.datetime(2018,1,1)) & (df['DateTime'] < datetime.datetime(2018,1,31))].drop_duplicates(subset='PatientID').shape[0]
# n_patients['02_18'] = df[(df['DateTime'] > datetime.datetime(2018,2,1)) & (df['DateTime'] < datetime.datetime(2018,2,28))].drop_duplicates(subset='PatientID').shape[0]
# n_patients['03_18'] = df[(df['DateTime'] > datetime.datetime(2018,3,1)) & (df['DateTime'] < datetime.datetime(2018,3,31))].drop_duplicates(subset='PatientID').shape[0]
# n_patients['04_18'] = df[(df['DateTime'] > datetime.datetime(2018,4,1)) & (df['DateTime'] < datetime.datetime(2018,4,30))].drop_duplicates(subset='PatientID').shape[0]
# n_patients['05_18'] = df[(df['DateTime'] > datetime.datetime(2018,5,1)) & (df['DateTime'] < datetime.datetime(2018,5,31))].drop_duplicates(subset='PatientID').shape[0]
# n_patients['06_18'] = df[(df['DateTime'] > datetime.datetime(2018,6,1)) & (df['DateTime'] < datetime.datetime(2018,6,30))].drop_duplicates(subset='PatientID').shape[0]
# n_patients['07_18'] = df[(df['DateTime'] > datetime.datetime(2018,7,1)) & (df['DateTime'] < datetime.datetime(2018,7,31))].drop_duplicates(subset='PatientID').shape[0]
# n_patients['08_18'] = df[(df['DateTime'] > datetime.datetime(2018,8,1)) & (df['DateTime'] < datetime.datetime(2018,8,31))].drop_duplicates(subset='PatientID').shape[0]
# n_patients['09_18'] = df[(df['DateTime'] > datetime.datetime(2018,9,1)) & (df['DateTime'] < datetime.datetime(2018,9,30))].drop_duplicates(subset='PatientID').shape[0]
# n_patients['10_18'] = df[(df['DateTime'] > datetime.datetime(2018,10,1)) & (df['DateTime'] < datetime.datetime(2018,10,31))].drop_duplicates(subset='PatientID').shape[0]
# n_patients['11_18'] = df[(df['DateTime'] > datetime.datetime(2018,11,1)) & (df['DateTime'] < datetime.datetime(2018,11,30))].drop_duplicates(subset='PatientID').shape[0]
# n_patients['12_18'] = df[(df['DateTime'] > datetime.datetime(2018,12,1)) & (df['DateTime'] < datetime.datetime(2018,12,31))].drop_duplicates(subset='PatientID').shape[0]
# #In[2019]
# n_patients['01_19'] = df[(df['DateTime'] > datetime.datetime(2019,1,1)) & (df['DateTime'] < datetime.datetime(2019,1,31))].drop_duplicates(subset='PatientID').shape[0]
# n_patients['02_19'] = df[(df['DateTime'] > datetime.datetime(2019,2,1)) & (df['DateTime'] < datetime.datetime(2019,2,28))].drop_duplicates(subset='PatientID').shape[0]
# n_patients['03_19'] = df[(df['DateTime'] > datetime.datetime(2019,3,1)) & (df['DateTime'] < datetime.datetime(2019,3,31))].drop_duplicates(subset='PatientID').shape[0]
# n_patients['04_19'] = df[(df['DateTime'] > datetime.datetime(2019,4,1)) & (df['DateTime'] < datetime.datetime(2019,4,30))].drop_duplicates(subset='PatientID').shape[0]
# n_patients['05_19'] = df[(df['DateTime'] > datetime.datetime(2019,5,1)) & (df['DateTime'] < datetime.datetime(2019,5,31))].drop_duplicates(subset='PatientID').shape[0]
# n_patients['06_19'] = df[(df['DateTime'] > datetime.datetime(2019,6,1)) & (df['DateTime'] < datetime.datetime(2019,6,30))].drop_duplicates(subset='PatientID').shape[0]
# n_patients['07_19'] = df[(df['DateTime'] > datetime.datetime(2019,7,1)) & (df['DateTime'] < datetime.datetime(2019,7,31))].drop_duplicates(subset='PatientID').shape[0]
# n_patients['08_19'] = df[(df['DateTime'] > datetime.datetime(2019,8,1)) & (df['DateTime'] < datetime.datetime(2019,8,31))].drop_duplicates(subset='PatientID').shape[0]
# n_patients['09_19'] = df[(df['DateTime'] > datetime.datetime(2019,9,1)) & (df['DateTime'] < datetime.datetime(2019,9,30))].drop_duplicates(subset='PatientID').shape[0]
# n_patients['10_19'] = df[(df['DateTime'] > datetime.datetime(2019,10,1)) & (df['DateTime'] < datetime.datetime(2019,10,31))].drop_duplicates(subset='PatientID').shape[0]
# n_patients['11_19'] = df[(df['DateTime'] > datetime.datetime(2019,11,1)) & (df['DateTime'] < datetime.datetime(2019,11,30))].drop_duplicates(subset='PatientID').shape[0]
# n_patients['12_19'] = df[(df['DateTime'] > datetime.datetime(2019,12,1)) & (df['DateTime'] < datetime.datetime(2019,12,31))].drop_duplicates(subset='PatientID').shape[0]
# #In[2020]
# n_patients['01_20'] = df[(df['DateTime'] > datetime.datetime(2020,1,1)) & (df['DateTime'] < datetime.datetime(2020,1,31))].drop_duplicates(subset='PatientID').shape[0]
# n_patients['02_20'] = df[(df['DateTime'] > datetime.datetime(2020,2,1)) & (df['DateTime'] < datetime.datetime(2020,2,28))].drop_duplicates(subset='PatientID').shape[0]
# n_patients['03_20'] = df[(df['DateTime'] > datetime.datetime(2020,3,1)) & (df['DateTime'] < datetime.datetime(2020,3,31))].drop_duplicates(subset='PatientID').shape[0]
# n_patients['04_20'] = df[(df['DateTime'] > datetime.datetime(2020,4,1)) & (df['DateTime'] < datetime.datetime(2020,4,30))].drop_duplicates(subset='PatientID').shape[0]
# n_patients['05_20'] = df[(df['DateTime'] > datetime.datetime(2020,5,1)) & (df['DateTime'] < datetime.datetime(2020,5,31))].drop_duplicates(subset='PatientID').shape[0]
# n_patients['06_20'] = df[(df['DateTime'] > datetime.datetime(2020,6,1)) & (df['DateTime'] < datetime.datetime(2020,6,30))].drop_duplicates(subset='PatientID').shape[0]
# n_patients['07_20'] = df[(df['DateTime'] > datetime.datetime(2020,7,1)) & (df['DateTime'] < datetime.datetime(2020,7,31))].drop_duplicates(subset='PatientID').shape[0]
# n_patients['08_20'] = df[(df['DateTime'] > datetime.datetime(2020,8,1)) & (df['DateTime'] < datetime.datetime(2020,8,31))].drop_duplicates(subset='PatientID').shape[0]
# n_patients['09_20'] = df[(df['DateTime'] > datetime.datetime(2020,9,1)) & (df['DateTime'] < datetime.datetime(2020,9,30))].drop_duplicates(subset='PatientID').shape[0]
# n_patients['10_20'] = df[(df['DateTime'] > datetime.datetime(2020,10,1)) & (df['DateTime'] < datetime.datetime(2020,10,31))].drop_duplicates(subset='PatientID').shape[0]
# n_patients['11_20'] = df[(df['DateTime'] > datetime.datetime(2020,11,1)) & (df['DateTime'] < datetime.datetime(2020,11,30))].drop_duplicates(subset='PatientID').shape[0]
# n_patients['12_20'] = df[(df['DateTime'] > datetime.datetime(2020,12,1)) & (df['DateTime'] < datetime.datetime(2020,12,31))].drop_duplicates(subset='PatientID').shape[0]
# #In[2021]
# n_patients['01_21'] = df[(df['DateTime'] > datetime.datetime(2021,1,1)) & (df['DateTime'] < datetime.datetime(2021,1,31))].drop_duplicates(subset='PatientID').shape[0]

# mask = df['DateTime'].isnull()
# n_patients['noDate'] = df[mask].drop_duplicates(subset='PatientID').shape[0]

# n_patients['noDate'] = df[mask].shape[0]




# # {'01_18': 11,
# #  '02_18': 0,
# #  '03_18': 0,
# #  '04_18': 0,
# #  '05_18': 0,
# #  '06_18': 0,
# #  '07_18': 0,
# #  '08_18': 0,
# #  '09_18': 0,
# #  '10_18': 0,
# #  '11_18': 0,
# #  '12_18': 0,
# #  '01_19': 2333,
# #  '02_19': 2050,
# #  '03_19': 2050,
# #  '04_19': 2062,
# #  '05_19': 2222,
# #  '06_19': 2147,
# #  '07_19': 1601,
# #  '08_19': 2123,
# #  '09_19': 2203,
# #  '10_19': 2107,
# #  '11_19': 2343,
# #  '12_19': 2080,
# #  '01_20': 10,
# #  '02_20': 3,
# #  '03_20': 4,
# #  '04_20': 2,
# #  '05_20': 4,
# #  '06_20': 4,
# #  '07_20': 0,
# #  '08_20': 1,
# #  '09_20': 0,
# #  '10_20': 1,
# #  '11_20': 4,
# #  '12_20': 0,
# #  '01_21': 1,
# #  'noDate': 9722}

# # In[n_patients]

# n_patients_per_folder = {}
# for i in range(1,10):
#     df = pd.read_csv(f'{tab_dir}healthy_{i}.csv',encoding= 'unicode_escape')
#     n_patients_per_folder[f"healthy_{i}"] = df.drop_duplicates(subset='PatientID').shape[0]
    
    
# for name in ['healthy_10_n','healthy_11_nn','healthy_12_nn','pos_n']:
#     df = pd.read_csv(f'{tab_dir}{name}.csv',encoding= 'unicode_escape')
#     n_patients_per_folder[name] = df.drop_duplicates(subset='PatientID').shape[0]


# # # {'healthy_1': 2641,
# # #  'healthy_2': 2312,
# # #  'healthy_3': 2251,
# # #  'healthy_4': 2370,
# # #  'healthy_5': 2465,
# # #  'healthy_6': 2306,
# # #  'healthy_7': 1793,
# # #  'healthy_8': 2256,
# # #  'healthy_9': 2441,
# # #  'healthy_10_n': 2430,
# # #  'healthy_11_nn': 2539,
# # #  'healthy_12_nn': 2267}

# In[counting positive patients]

df = pd.read_csv(f'{tab_dir}pos_nn.csv',encoding= 'unicode_escape')
#In[Convert time and date to datetime for efficient access]
time_k = 'InstanceCreationTime'
date_k = 'InstanceCreationDate'
df['DateTime'] = df[date_k] + ' ' +  df[time_k]
#date_time_m = df['DateTime'].isnull()
df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y%m%d %H:%M:%S')

n_patients_pos = {}
#In[2020]
n_patients_pos['01_20'] = df[(df['DateTime'] > datetime.datetime(2020,1,1)) & (df['DateTime'] < datetime.datetime(2020,1,31))].drop_duplicates(subset='PatientID').shape[0]
n_patients_pos['02_20'] = df[(df['DateTime'] > datetime.datetime(2020,2,1)) & (df['DateTime'] < datetime.datetime(2020,2,28))].drop_duplicates(subset='PatientID').shape[0]
n_patients_pos['03_20'] = df[(df['DateTime'] > datetime.datetime(2020,3,1)) & (df['DateTime'] < datetime.datetime(2020,3,31))].drop_duplicates(subset='PatientID').shape[0]
n_patients_pos['04_20'] = df[(df['DateTime'] > datetime.datetime(2020,4,1)) & (df['DateTime'] < datetime.datetime(2020,4,30))].drop_duplicates(subset='PatientID').shape[0]
n_patients_pos['05_20'] = df[(df['DateTime'] > datetime.datetime(2020,5,1)) & (df['DateTime'] < datetime.datetime(2020,5,31))].drop_duplicates(subset='PatientID').shape[0]
n_patients_pos['06_20'] = df[(df['DateTime'] > datetime.datetime(2020,6,1)) & (df['DateTime'] < datetime.datetime(2020,6,30))].drop_duplicates(subset='PatientID').shape[0]
n_patients_pos['07_20'] = df[(df['DateTime'] > datetime.datetime(2020,7,1)) & (df['DateTime'] < datetime.datetime(2020,7,31))].drop_duplicates(subset='PatientID').shape[0]
n_patients_pos['08_20'] = df[(df['DateTime'] > datetime.datetime(2020,8,1)) & (df['DateTime'] < datetime.datetime(2020,8,31))].drop_duplicates(subset='PatientID').shape[0]
n_patients_pos['09_20'] = df[(df['DateTime'] > datetime.datetime(2020,9,1)) & (df['DateTime'] < datetime.datetime(2020,9,30))].drop_duplicates(subset='PatientID').shape[0]
n_patients_pos['10_20'] = df[(df['DateTime'] > datetime.datetime(2020,10,1)) & (df['DateTime'] < datetime.datetime(2020,10,31))].drop_duplicates(subset='PatientID').shape[0]
n_patients_pos['11_20'] = df[(df['DateTime'] > datetime.datetime(2020,11,1)) & (df['DateTime'] < datetime.datetime(2020,11,30))].drop_duplicates(subset='PatientID').shape[0]
n_patients_pos['12_20'] = df[(df['DateTime'] > datetime.datetime(2020,12,1)) & (df['DateTime'] < datetime.datetime(2020,12,31))].drop_duplicates(subset='PatientID').shape[0]
#In[2021]
n_patients_pos['01_21'] = df[(df['DateTime'] > datetime.datetime(2021,1,1)) & (df['DateTime'] < datetime.datetime(2021,1,31))].drop_duplicates(subset='PatientID').shape[0]
n_patients_pos['02_21'] = df[(df['DateTime'] > datetime.datetime(2121,2,1)) & (df['DateTime'] < datetime.datetime(2121,2,28))].drop_duplicates(subset='PatientID').shape[0]
n_patients_pos['03_21'] = df[(df['DateTime'] > datetime.datetime(2121,3,1)) & (df['DateTime'] < datetime.datetime(2121,3,31))].drop_duplicates(subset='PatientID').shape[0]
n_patients_pos['04_21'] = df[(df['DateTime'] > datetime.datetime(2121,4,1)) & (df['DateTime'] < datetime.datetime(2121,4,30))].drop_duplicates(subset='PatientID').shape[0]
n_patients_pos['05_21'] = df[(df['DateTime'] > datetime.datetime(2121,5,1)) & (df['DateTime'] < datetime.datetime(2121,5,31))].drop_duplicates(subset='PatientID').shape[0]
n_patients_pos['06_21'] = df[(df['DateTime'] > datetime.datetime(2121,6,1)) & (df['DateTime'] < datetime.datetime(2121,6,30))].drop_duplicates(subset='PatientID').shape[0]

mask = df['DateTime'].isnull()
n_patients_pos['noDate'] = df[mask].drop_duplicates(subset='PatientID').shape[0]

# In[]
# # {'01_20': 54,
# #  '02_20': 47,
# #  '03_20': 50,
# #  '04_20': 68,
# #  '05_20': 54,
# #  '06_20': 66,
# #  '07_20': 52,
# #  '08_20': 51,
# #  '09_20': 62,
# #  '10_20': 74,
# #  '11_20': 72,
# #  '12_20': 93,
# #  '01_21': 96,
# #  '02_21': 0,
# #  '03_21': 0,
# #  '04_21': 0,
# #  '05_21': 0,
# #  '06_21': 0,
# #  'noDate': 294}

# In[Counting studies over months for positive patients]

df = pd.read_csv(f'{tab_dir}pos_nn.csv',encoding= 'unicode_escape')
#In[Convert time and date to datetime for efficient access]
time_k = 'InstanceCreationTime'
date_k = 'InstanceCreationDate'
df['DateTime'] = df[date_k] + ' ' +  df[time_k]
#date_time_m = df['DateTime'].isnull()
df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y%m%d %H:%M:%S')


n_studies = {}
for month in pos_months[:-1]: #['01_20','02_20','03_20','04_20','05_20','06_20','07_20','08_20','09_20','10_20','11_20','12_20','01_21','02_21','03_21','04_21','05_21','06_21']:
    df_month = df[time_masks[month]]
    df_p_sorted = df_month.groupby('PatientID').apply(
        lambda x: (x.sort_values(by=['DateTime'], ascending=True)))
    # Count the number of studies]
    patient_ids = df_p_sorted['PatientID'].unique()
    num_studies_l = []
    for patient in patient_ids:
        patient_mask = df_p_sorted['PatientID']==patient
        date_times = df_p_sorted[patient_mask]['DateTime']
        print(date_times)
        date_time0 = date_times[0]
        study_counter = 1
        for date_time in date_times[1:]:
            try:
                time_diff = date_time-date_time0
                if time_diff.total_seconds()/3600>2:
                    study_counter += 1
                    date_time0 = date_time
                else:
                    pass
            except:
                print('NaT')
        num_studies_l.append(study_counter)
    n_studies[month] = sum(num_studies_l)
    print('end')

n_studies['noDate'] = df[time_masks['noDate']].drop_duplicates(subset='StudyInstanceUID').shape[0]
# {'01_20': 56,
#  '02_20': 51,
#  '03_20': 60,
#  '04_20': 72,
#  '05_20': 59,
#  '06_20': 72,
#  '07_20': 56,
#  '09_20': 69,
#  '10_20': 88,
#  '11_20': 80,
#  '12_20': 115,
#  '01_21': 112,
#  '02_21': 0,
#  '03_21': 0,
#  '04_21': 0,
#  '05_21': 0,
#  '06_21': 0}

# In[Scans positive patients per month]

n_volumes = {}
for month in pos_months:
    n_volumes[month] = df[time_masks[month]].drop_duplicates(subset='SeriesInstanceUID').shape[0]
# # {'01_20': 593,
# #  '02_20': 433,
# #  '03_20': 659,
# #  '04_20': 781,
# #  '05_20': 616,
# #  '06_20': 703,
# #  '07_20': 556,
# #  '08_20': 490,
# #  '09_20': 606,
# #  '10_20': 938,
# #  '11_20': 698,
# #  '12_20': 1108,
# #  '01_21': 1067,
# #  '02_21': 0,
# #  '03_21': 0,
# #  '04_21': 0,
# #  '05_21': 0,
# #  '06_21': 0,
# #  'noDate': 1890}

# In[Studies for negative patients]

n_studies = {}
for month in neg_months[:-1]:
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
            for date_time in date_times[1] :#[indexes[1]:]:
                try:
                    time_diff = date_time-date_time0
                    if time_diff.total_seconds()/3600>2:
                        study_counter += 1
                        date_time0 = date_time
                    else:
                        pass
                except:
                    print('NaT')
        num_studies_l.append(study_counter)
    n_studies[month] = sum(num_studies_l)
    print('end')


# {'01_18': 11, '02_18': 0, '03_18': 0, '04_18': 0, '05_18': 0, '06_18': 0, '07_18': 0, '08_18': 0, '09_18': 0, '10_18': 0, '11_18': 0, '12_18': 0, '01_19': 2333, '02_19': 2050, '03_19': 2050, '04_19': 2062, '05_19': 2222, '06_19': 2147, '07_19': 1601, '08_19': 2123, '09_19': 2203, '10_19': 2107, '11_19': 2343, '12_19': 2080, '01_20': 10, '02_20': 3, '03_20': 4, '04_20': 2, '05_20': 4, '06_20': 4, '07_20': 0, '08_20': 1, '09_20': 0, '10_20': 1, '11_20': 4, '12_20': 0, '01_21': 1, '02_21': 0, '03_21': 0, '04_21': 0, '05_21': 0, '06_21': 0}
# 'nodate':11499

# In[Counting studies for postive patients]


df = pd.read_csv(f'{tab_dir}pos_nn.csv',encoding= 'unicode_escape')
#In[Convert time and date to datetime for efficient access]
time_k = 'InstanceCreationTime'
date_k = 'InstanceCreationDate'
df['DateTime'] = df[date_k] + ' ' +  df[time_k]
#date_time_m = df['DateTime'].isnull()
df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y%m%d %H:%M:%S')

df_month = df.drop_duplicates(subset='StudyInstanceUID')
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
        print('mes d1')
        for date_time in date_times[1:] :#[indexes[1]:]:
            try:
                time_diff = date_time-date_time0
                if time_diff.total_seconds()/3600>2:
                    study_counter += 1
                    date_time0 = date_time
                else:
                    pass
            except:
                study_counter +=1
                print('NaT')
    num_studies_l.append(study_counter)