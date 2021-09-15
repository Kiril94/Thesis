# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 11:41:20 2021

@author: klein
"""
import numpy as np
import itertools
import pandas as pd


time_k = 'InstanceCreationTime'
date_k = 'InstanceCreationDate'

def only_first_true(a, b):
    """takes two binary arrays
    and returns True
    where only the el. of the first array is true
    if only the second or both are true returns false"""
    return a&np.logical_not(a&b)

def mask_sequence_type(df, str_, key='SeriesDescription'):
    """Checks all the values in df/groupby
    in the column key (SeriesDescription by default),
    if they contain the string str_. Returns a mask.
    """
    try:
        mask = df[key].str.contains(str_, na=False)
    except:
        mask = df.apply(lambda x: x[key].str.contains(str_, na=False))
    return mask

def check_tags(df, tags, key='SeriesDescription'):
    """calls mask_sequence type for a list of tags and combines
    the masks with or"""
    masks = []
    for tag in tags:
        masks.append(mask_sequence_type(df, tag, key))
    mask = masks[0]
    for i in range(1, len(masks)):
        mask = mask | masks[i]
    return mask

def group_small(dict_, threshold, keyword='other'):
    """Takes a dictionary and sums all values that are smaller than threshold
    the result is stored under the key keyword. Useful for pie charts."""
    newdic={}
    for key, group in itertools.groupby(dict_, lambda k: keyword \
                                        if (dict_[k]<threshold) else k):
         newdic[key] = sum([dict_[k] for k in list(group)]) 
    return newdic

def count_number_of_studies(df, threshold=2):
    """Counts the number of studies in a dataframe, studies are considered
    separate if they are at least threshold (default 2) hours apart
    returns a list with the number of studies where every entry represents a
    patient"""
    
    df_sorted = df.groupby('PatientID').apply(
        lambda x: (x.sort_values(by=['DateTime'], ascending=True)))
    
    patient_ids = df_sorted['PatientID'].unique()
    num_studies_l = []
    for patient in patient_ids:
        patient_mask = df_sorted['PatientID']==patient
        date_times = df_sorted[patient_mask]['DateTime']
        date_time0 = date_times[0]
        study_counter = 1
        for date_time in date_times[1:]:
            if pd.isnull(date_time):
                continue
            try:
                time_diff = date_time-date_time0
                if time_diff.total_seconds()/3600>threshold:
                    study_counter += 1
                    date_time0 = date_time
                else:
                    pass
            except:
                print('An error occured')
        num_studies_l.append(study_counter)
    return num_studies_l

def time_between_studies(df, threshold=2):
    """Returns a list with the times between consecutive studies in a dataframe, 
    studies are considered separate if they are at least threshold (default 2) 
    hours apart
    """
    df_sorted = df.groupby('PatientID').apply(
        lambda x: (x.sort_values(by=['DateTime'], ascending=True)))
    patient_ids = df_sorted['PatientID'].unique()
    time_diff_l = []
    for patient in patient_ids:
        patient_mask = df_sorted['PatientID']==patient
        date_times = df_sorted[patient_mask]['DateTime']
        date_time0 = date_times[0]
        for date_time in date_times[1:]:
            if pd.isnull(date_time):
                continue
            else:
                try:
                    time_diff = date_time-date_time0
                    if time_diff.total_seconds()/3600>threshold:
                        time_diff_l.append(time_diff.total_seconds()/3600)
                except:
                    print('An error occured')
                date_time0 = date_time
    return time_diff_l

def add_datetime(df):
    df['DateTime'] = df[date_k] + ' ' +  df[time_k]
    df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y%m%d %H:%M:%S')
    return df