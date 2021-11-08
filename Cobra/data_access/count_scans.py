#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 11:13:24 2021

@author: neus
"""

from statistics_functions import * 
from functools import reduce
import datetime

#Read data

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