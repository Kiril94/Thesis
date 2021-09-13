# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 17:21:34 2021

@author: klein
"""
import os
import json
from glob import iglob
import pandas as pd


def literal_converter(val):
    # replace first val with '' or some other null identifier if required
    try:
        result = None if (val == '' or val=='NONE') else eval(val)
    except:
        result = [val]
    return result

def load_scan_csv(csv_path):
    """Returns a dataframe
    Takes into account that some columns store lists."""
    try:
        df = pd.read_csv(
            csv_path, encoding='unicode_escape',
            converters={
                k: literal_converter for k in\
                    ['ImageType', 'SequenceVariant', 'ScanOptions',
                     'PixelSpacing']})
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

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
def list_subdir(dir_):
    return [os.path.join(dir_, x) for x in os.listdir(dir_)]

def get_json(path):
    """Returns data, contained in a json file under path."""
    with open(path, 'r') as f:
    	data = json.load(f)
    return data

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
