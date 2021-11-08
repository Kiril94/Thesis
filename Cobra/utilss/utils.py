# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 17:21:34 2021

@author: klein
"""
import os
import json
import time
from glob import iglob
import pandas as pd
from data_access import load_data_tools as ld
import csv
import datetime
import numpy as np

def read_pos_data():
    tab_dir = '/home/neus/Documents/09.UCPH/MasterThesis/github/Thesis/Cobra/tables/pos_nn.csv'
    df = pd.read_csv(tab_dir,encode='unicode_escape')
    
    #In[Convert time and date to datetime for efficient access]
    time_k = 'InstanceCreationTime'
    date_k = 'InstanceCreationDate'
    df['DateTime'] = df[date_k] + ' ' +  df[time_k]
    #date_time_m = df['DateTime'].isnull()
    df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y%m%d %H:%M:%S')
    
    return df     

def read_whole_data():
    tab_dir = '/home/neus/Documents/09.UCPH/MasterThesis/github/Thesis/Cobra/tables/'
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
    
    return df 

def write_csv(csv_path, patient_list, append=False):
    """This function takes a patient list with patient directories
    and writes the relevant tags of the dicom header to csv_path
    if append==False existing csv files are overwritten."""
    if append:
        mode = 'a'
    else:
        mode = 'w'
    csv_columns = [x[0] for x in ld.get_scan_key_list()]
    pat_counter = 0 
    with open(csv_path, mode, newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        if mode == 'w': 
            writer.writeheader()
        start = time.time()
        print('Start writing')
        for pat in patient_list:
            pat_counter += 1
            scan_directories = ld.Patient(pat).get_scan_directories()
            for scan_dir in scan_directories:
                try:
                    data = ld.get_scan_dictionary(scan_dir, 
                                                  reconstruct_3d=False)    
                except:
                    print("Sleep for 5s, maybe connection is lost")
                    time.sleep(5)
                    data = ld.get_scan_dictionary(scan_dir, 
                                                  reconstruct_3d=False)
                try:
                    writer.writerow(data)  
                except IOError:
                    print("I/O error")
                print('.', end='')
            if pat_counter%100==0:
                print(f"{pat_counter} patients written")
                print(f"{(time.time()-start)/60} min passed")
            print(f"{pat} stored to csv")
        stop = time.time()
    print(f"the conversion took {(stop-start)/3600} h")

        
def literal_converter(val):
    """Helper function for load_scan_csv"""
    try:
        result = None if (val == '' or val=='NONE') else eval(val)
    except:
        result = [val]
    return result

def date_time_converter(val):
    try:
        result = None if (val == '' or val=='NONE') \
            else datetime.datetime.strptime(val,'%Y-%m-%d %H:%M:%S')
    except:
        result = val
    return result

def load_scan_csv(csv_path):
    """Returns a dataframe
    Takes into account that some columns store lists."""
    try:
        df = pd.read_csv(
            csv_path, encoding='unicode_escape',
            converters={**{
                k: literal_converter for k in\
                    ['ImageType', 'SequenceVariant', 'ScanOptions',
                     'PixelSpacing']},**{'DateTime':date_time_converter}})
    except: 
        df = pd.read_csv(
            csv_path, encoding='unicode_escape',
            converters={
                k: literal_converter for k in\
                    ['ImageType', 'SequenceVariant, ScanOptions']})
        print('Once PixelSpacing is added the try-except statement should be removed')
    return df

def count_subdirectories(dir_, level=1, count_all=True):
    """Counts all folders on the specified level.
    if count_all==True: also files are counter"""
    dir_str = str(dir_)
    for _ in range(level):
        dir_str = dir_str + "/*"
    if not(count_all):
        result = sum(1 for x in iglob(dir_str) if os.path.isdir(x))
    else:
        result = sum(1 for x in iglob(dir_str))
    return result

    
def list_subdir(dir_):
    return [os.path.join(dir_, x) for x in os.listdir(dir_)]

def get_json(path):
    """Returns data, contained in a json file under path."""
    with open(path, 'r') as f:
    	data = json.load(f)
    return data

def get_running_time(start):
    m, s = divmod(time.time()-start , 60)
    h, m = divmod(m, 60)
    return f'[{h:2.0f}h{m:2.0f}m{s:2.0f}s]' 


def create_dictionary(keys, values):
    result = {} # empty dictionary
    for key, value in zip(keys, values):
        result[key] = value
    return result



def get_size(start_path = '.', unit='M'):
    """Gives size in bytes"""
    if unit=='':
        divider = 1
    elif unit=='M':
        divider = 1000
    elif unit=='G':
        divider = 1e6
    else:
        print(f"unit {unit} unknown")
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size/divider
