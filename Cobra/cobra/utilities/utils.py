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
from access_sif_data import load_data_tools as ld
import csv
from datetime import datetime as dt
from pathlib import Path
import logging



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

def directories_to_csv(csv_path, main_dir, append=False):
    """This function writes all the series directories in main_dir to csv."""
    if append:
        mode = 'a'
    else:
        mode = 'w'
    fieldnames = ['SeriesInstanceUID', 'Directory']
    with open(csv_path, mode, newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if mode == 'w': 
            writer.writeheader()
        start = time.time()
        counter = 0
        print(f'Start writing from {main_dir}')
        for scan_dir in iglob(f"{main_dir}/*/*/MR/*"):
            SID = Path(scan_dir).parts[-1]
            scan_dir_no_drive = os.path.splitdrive(scan_dir)[1][2:]
            data = {'SeriesInstanceUID':SID,
                    'Directory':scan_dir_no_drive}
            try:
                writer.writerow(data)  
            except IOError:
                print("I/O error")
            print('.', end='')
            counter+=1
            if counter%1000==0:
                print(f"{counter} series directories written")
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
            else dt.strptime(val,'%Y-%m-%d %H:%M:%S')
    except:
        result = val
    return result

def date_converter(val):
    try:
        return dt.strptime(val, "%Y-%m-%d").date()
    except:
        return pd.NaT
def time_converter(val):
    try:
        return dt.strptime(val, "%H:%M:%S").time()
    except:
        return pd.NaT

def load_scan_csv(csv_path):
    """Returns a dataframe
    Takes into account that some columns store lists."""
    try:
        df = pd.read_csv(
            csv_path, 
            converters={
                **{
                k: literal_converter for k in\
                    ['ScanningSequence','ImageType', 'SequenceVariant',
                     'ScanOptions',
                     'PixelSpacing']},
                **{'DateTime':date_time_converter, 
                'InstanceCreationDate':date_converter, 
                'InstanceCreationTime':time_converter,}, 
                    },
                     )
    except: 
        df = pd.read_csv(
            csv_path, 
            converters={
                **{k: literal_converter for k in\
                    ['ScanningSequence', 'ImageType', 'SequenceVariant, ScanOptions']},
                    **{'InstanceCreationDate':date_converter, 
                     'InstanceCreationTime':time_converter,}},
                    )
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


def get_json(path):
    """Returns data, contained in a json file under path."""
    with open(path, 'r') as f:
    	data = json.load(f)
    return data

def get_running_time(start):
    m, s = divmod(time.time()-start , 60)
    h, m = divmod(m, 60)
    return f'[{h:2.0f}h{m:2.0f}m{s:2.0f}s]' 

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

def convert_to_sparse_pandas(df, exclude_columns=[]):
    """
    Converts columns of a data frame into SparseArrays and returns the data frame with transformed columns.
    Use exclude_columns to specify columns to be excluded from transformation.
    :param df: pandas data frame
    :param exclude_columns: list
        Columns not be converted to sparse
    :return: pandas data frame
    https://towardsdatascience.com/working-with-sparse-data-sets-in-pandas-and-sklearn-d26c1cfbe067
    """
    df = df.copy()
    exclude_columns = set(exclude_columns)

    for (columnName, columnData) in df.iteritems():
        if columnName in exclude_columns:
            continue
        df[columnName] = pd.SparseArray(columnData.values, dtype='uint8')

    return df
