from datetime import datetime as dt
from os.path import join, split
import os

from sympy import re
from utilities.basic import list_subdir
import numpy as np
from pydicom import dcmread
import multiprocessing
import json
import pickle
import time
import sys
from pathlib import Path
from functools import partial
import pandas as pd
import warnings



def get_value_from_header(dcm_dir, key):
    dcm = dcmread(dcm_dir)
    return dcm[key].value
def get_slice_location(dcm_dir):
    dcm = dcmread(dcm_dir)
    return dcm['SliceLocation'].value
def get_image_orientation(dcm_dir):
    dcm = dcmread(dcm_dir)
    return dcm['ImageOrientationPatient'].value
def get_image_position(dcm_dir):
    dcm = dcmread(dcm_dir)
    return dcm['ImagePositionPatient'].value

def compute_dist_from_slice_location(dcm_dirs, aggregating_func, test=False):
    """Uses the location of slices to compute the distance between them.
    A maximum of floor(num_slices/2)-1 slices can be missing 
    to ensure a correct coputation of slices"""
    locations = []
    n_missing = 0
    for dcm_dir in dcm_dirs:
        if n_missing == np.floor(len(dcm_dirs)/2):
            return -1
        try:
            location = get_value_from_header(dcm_dir, 'SliceLocation')
            locations.append(location) 
        except:
            n_missing+=1
    locations = np.array(locations)
    locations = locations[~np.isnan(locations)]
    
    if len(locations)<np.floor(len(dcm_dirs)/2):
        return -1
    else:
        loc_diff = np.ediff1d(np.sort(locations))
        if test:
            print(loc_diff)
        slice_dist = aggregating_func(loc_diff)
        if slice_dist<0 or slice_dist>20:
            return -1
        else:
            return slice_dist

def get_value_from_header_except(dcm_dirs, value, file_num=0):
    if file_num==int(len(dcm_dirs)/2):
        return None 
    try:
        cosines = get_value_from_header(dcm_dirs[file_num], value)
        return cosines
    except:
        file_num+=1
        cosines = get_value_from_header_except(dcm_dirs, value, file_num)
        return cosines

def compute_dist_from_img_pos_and_orientation(dcm_dirs, aggregation_func, test=False):
    cosines = get_value_from_header_except(dcm_dirs, value='ImageOrientationPatient', file_num=0)
    if cosines is None:
        return -1
    normal = np.cross(cosines[:3], cosines[3:])
    distances = []
    n_missing = 0
    for dcm_dir in dcm_dirs:
        if n_missing == np.floor(len(dcm_dirs)/2):
            return -1
        try:
            ipp = get_value_from_header(dcm_dir, 'ImagePositionPatient')
            dist = np.sum(normal*ipp)
            distances.append(dist)
        except:
            n_missing+=1
    dist_arr = np.array(distances)
    dist_arr = dist_arr[~np.isnan(dist_arr)]
    if len(dist_arr)<np.floor(len(dcm_dirs)/2):
        return -1
    else:
        dist_arr_sorted = np.sort(dist_arr)
        dist_between_slices = np.ediff1d(dist_arr_sorted)
        if test:
            print(dist_between_slices)
        slice_dist = aggregation_func(dist_between_slices)
        if slice_dist<0 or slice_dist>20:
            return -1
        else:
            return slice_dist

def write_dist_to_file(write_file, series_dir, dist):
    with open(write_file,"a+") as file:
        file.write(f"{series_dir} {dist}\n")

def get_distances_for_series(series_dir, 
            aggregation_func, write_file_dir, test=False):
    if not test:
        current_proc = multiprocessing.current_process()    
        current_proc_id = int(current_proc._identity[0])
        write_file = join(write_file_dir, f'slice_dist_{current_proc_id}.txt')
    sid = split(series_dir)[1]
    try:
        dcm_dirs = list_subdir(series_dir)
    except Exception as e:
        print("ERROR : "+str(e))
    dist = compute_dist_from_slice_location(dcm_dirs, aggregation_func, test)
    if dist==-1:
        dist = compute_dist_from_img_pos_and_orientation(
                dcm_dirs, aggregation_func, test)
        if dist==-1:
            sys.stdout.flush()
            print('x', end='')
            print(sid)
        else: 
            if not test:
                sys.stdout.flush()
                print('.', end='')
                write_dist_to_file(write_file, sid, dist)
            else:
                print('worked')
                print(dist)
    else:
        if not test:
            sys.stdout.flush()
            print('.', end='')
            write_dist_to_file(write_file, sid, dist)
        else:
            print('worked')
            print(dist)
        
    
def merge_output(write_file_dir):
    dist_files = [f for f in os.listdir(write_file_dir) \
        if f.startswith("slice_dist")]
    dist_paths = [join(write_file_dir, f) for f in dist_files]
    string = ''
    for dist_path in dist_paths:
        with open(dist_path, 'r') as f:
            text = f.read()
        string+=text
    with open(join(write_file_dir, 'all_distances.txt'), 'a+') as f:
        f.write(string)

def get_existing_sids(all_dist_path):
    df = pd.read_csv(all_dist_path, header=None, delimiter=' ', 
            names=['SeriesInstanceUID','DistanceBetweenSlices'])
    ex_sids = df.drop_duplicates().SeriesInstanceUID
    return ex_sids

def remove_existing_sids(sids, all_dist_path):
    ex_sids = get_existing_sids(all_dist_path)
    rest_sids = list(set(sids).difference(set(ex_sids)))
    return rest_sids

def get_rest_sids(sids_path, all_dist_path):
    with open(sids_path, 'rb') as f:
        sids = pickle.load(f)
    rest_sids = remove_existing_sids(sids, all_dist_path)
    return rest_sids

def save_distance_between_slices(sids, volume_dir_dic, num_of_procs=8, 
    write_file_dir="", aggregation_func=np.median, test=False):
    """Save missing tags to text."""
    series_dirs = [volume_dir_dic[sid] for sid in sids]
    get_distances_for_series_partial = partial(
        get_distances_for_series,
        aggregation_func=aggregation_func, write_file_dir=write_file_dir, test=test)     
    if not test:
        with multiprocessing.Pool(num_of_procs) as pool:
                    pool.map(get_distances_for_series_partial, 
                            series_dirs)
    else:
        for series_dir in series_dirs:
            get_distances_for_series(
                series_dir, aggregation_func=aggregation_func, 
                write_file_dir=write_file_dir, test=test)

def main(num_of_procs, write_file_dir, volume_dir_dic, sids, aggregation_func=np.median, test=False):
    save_distance_between_slices(sids, volume_dir_dic,
            num_of_procs, write_file_dir, aggregation_func, 
            test=test)
    if not test:
        merge_output(write_file_dir)
    return 0


script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parent
table_dir = join(base_dir, 'data', 'tables')
dicom_base_dir = "F:/CoBra/Data/dcm"
write_file_dir = join(base_dir, 'data/t1_cross/distance_between_slices')

with open(join(table_dir, "disk_series_directories.json"), "r") as json_file:
    volume_dir_dic = json.load(json_file)

if __name__=="__main__":
    compute_distance = True
    if compute_distance:
        rest_sids = sorted(get_rest_sids(
            join(base_dir, 'data/t1_longitudinal/pairs_3dt1_long_sids.pkl'),
            join(write_file_dir, 'all_distances.txt')))
        print(len(rest_sids), 'sids before removing non-downloaded volumes')
        print('Take only downloaded volumes')
        with open(join(dicom_base_dir, 'volume_log.txt'), 'r') as f:
            dwnld_sids = [line.strip() for line in f]
        rest_sids = list(set(rest_sids).intersection(set(dwnld_sids)))
        print(len(rest_sids), 'sids')
        warnings.warn("Don't forget to update volume_dir_dic")
        print('Take only sids that are in volume_dir_dic')
        rest_sids = list(set(rest_sids).intersection(set(volume_dir_dic.keys())))
        test=False
        if test:
            main(1, write_file_dir='', volume_dir_dic=volume_dir_dic, sids=rest_sids[:2], test=True)
        else:
            start=time.time()
            print("Compute distance between slices for ", len(rest_sids), 'volumes')
            return_status = main(10, write_file_dir, volume_dir_dic, rest_sids)
            print('finished')
            print(f'status: {return_status}')
            print(f'Time (min): {(time.time()-start)/60:.3f}')
            print(dt.now())
    else:
        all_dist_path =  join(write_file_dir, 'all_distances.txt')
        df_dist = pd.read_csv(all_dist_path, header=None, delimiter=' ', 
                names=['SeriesInstanceUID','DistanceBetweenSlices'])        
        df_dist = df_dist.drop_duplicates()
        dfc = pd.read_csv(join(table_dir, 'neg_pos_clean.csv'))
        print("length before merging", dfc.SeriesInstanceUID.nunique())
        dfc_new = pd.merge(dfc, df_dist, on='SeriesInstanceUID', how='left')
        print("length after merging", dfc_new.SeriesInstanceUID.nunique())
        print(dfc_new.head())
        print(dfc_new[~dfc_new.DistanceBetweenSlices.isna()])
        dfc_new.to_csv(join(table_dir,"neg_pos_clean.csv"), index=False, header=True)
# For n images, how many images m can be missing
# in order to compute the slice distance: 
# m<floor(n)/2