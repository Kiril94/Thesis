from datetime import datetime as dt
from os.path import join, split, normpath
import os
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

def compute_dist_from_slice_location(dcm_dirs, aggregating_func):
    """Uses the location of slices to compute the distance between them.
    A maximum of floor(num_slices/2)-1 slices can be missing 
    to ensure a correct coputation of slices"""
    locations = []
    n_missing = 0
    for dcm_dir in dcm_dirs:
        if n_missing == np.floor(len(dcm_dirs)/2):
            break
        try:
            location = get_value_from_header(dcm_dir, 'SliceLocation')
            locations.append(location) 
        except:
            n_missing+=1
            continue
    locations = np.array(locations)
    loc_diff = np.ediff1d(np.sort(locations))
    slice_dist = aggregating_func(loc_diff)
    return slice_dist

def get_value_from_header_except(dcm_dirs, file_num):
    if file_num==int(len(dcm_dirs)/2):
        return None 
    try:
        cosines = get_value_from_header(dcm_dirs[file_num], 
            'ImageOrientationPatient')
        return cosines
    except:
        file_num+=1
        cosines = get_value_from_header(dcm_dirs[file_num],
            'ImageOrientationPatient')
        return cosines

def compute_dist_from_img_pos_and_orientation(dcm_dirs, aggregating_func):
    cosines = get_value_from_header_except(dcm_dirs, file_num=0)
    assert cosines is not None, "cosines is None"
    normal = np.cross(cosines[:3], cosines[3:])
    distances = []
    n_missing = 0
    for dcm_dir in dcm_dirs:
        if n_missing == np.floor(len(dcm_dirs)/2):
            break
        try:
            ipp = get_value_from_header(dcm_dir, 'ImagePositionPatient')
            dist = np.sum(normal*ipp)
            distances.append(dist)
        except:
            n_missing+=1
            continue  
    dist_arr = np.array(distances)
    dist_arr_sorted = np.sort(dist_arr)
    dist_between_slices = np.ediff1d(dist_arr_sorted)
    return dist_between_slices

def write_dist_to_file(write_file, series_dir, dist):
    with open(write_file,"a+") as file:
        file.write(f"{series_dir} {dist}\n")

def get_distances_for_series(series_dir, 
            aggregation_func, write_file_dir):
    current_proc = multiprocessing.current_process()    
    current_proc_id = int(current_proc._identity[0])
    write_file = join(write_file_dir, f'slice_dist_{current_proc_id}.txt')
    sid = split(series_dir)[1]
    try:
        dcm_dirs = list_subdir(series_dir)
    except Exception as e:
        print("ERROR : "+str(e))
    sys.stdout.flush()
    try:
        dist = compute_dist_from_slice_location(dcm_dirs, aggregation_func)
        write_dist_to_file(write_file, sid, dist)
    except:
        try:
            dist = compute_dist_from_img_pos_and_orientation(
                dcm_dirs, aggregation_func)
            write_dist_to_file(write_file_dir, sid, dist)
        except: pass
    
def save_distance_between_slices(sids, 
    dicoms_base_dir='F:/', num_of_procs=8, 
    write_file_dir="", aggregation_func=np.min):
    """Save missing tags to text."""
    with open(join(base_dir, 'data/t1_longitudinal/sif_dir.json'), 'r') as fp:
        sif_dir_dic = json.load(fp)
    patient_dirs = [join(*normpath(sif_dir_dic[sid]).split(os.sep)[:2]) for sid in sids]
    series_dirs = [join(dicoms_base_dir, patient_dirs[i], sid ) \
                for i, sid in enumerate(sids)]
    get_distances_for_series_partial = partial(
        get_distances_for_series,
        aggregation_func=aggregation_func, write_file_dir=write_file_dir) 
    with multiprocessing.Pool(num_of_procs) as pool:
                pool.map(get_distances_for_series_partial, 
                        series_dirs)

def merge_output(write_file_dir):
    dist_files = [f for f in os.listdir(write_file_dir) \
        if f.startswith("slice_dist")]
    dist_paths = [join(write_file_dir, f) for f in dist_files]
    string = '\n'
    for dist_path in dist_paths:
        with open(dist_path, 'r') as f:
            text = f.read() + '\n'
        string+=text
    with open(join(write_file_dir, 'all_distances.txt'), 'a+') as f:
        f.write(string)

def get_existing_sids(all_dist_path):
    df = pd.read_csv(all_dist_path, sep=" ", header=None)
    sids = df.iloc[:,0]
    return sids
def remove_existing_sids(sids, all_dist_path):
    ex_sids = get_existing_sids(all_dist_path)
    rest_sids = list(set(sids).difference(set(ex_sids)))
    return rest_sids
def get_rest_sids(sids_path, all_dist_path):
    with open(sids_path, 'rb') as f:
        sids = pickle.load(f)
    sids_rest = remove_existing_sids(sids, all_dist_path)
    return sids_rest
def main(num_of_procs, write_file_dir, sids, aggregation_func=np.min):
    save_distance_between_slices(sids, 
            dicoms_base_dir=dicom_base_dir, num_of_procs=num_of_procs, 
            aggregation_func=aggregation_func, write_file_dir=write_file_dir)
    merge_output(write_file_dir)
    return 0

script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parent
dicom_base_dir = "F:/CoBra/Data/dcm"
write_file_dir = join(base_dir, 
        'data/t1_longitudinal/distance_between_slices')

sids_rest = get_rest_sids(join(base_dir, 'data/t1_longitudinal/pairs_3dt1_longitudinal_study.pkl'),
                 join(write_file_dir, 'all_distances.txt'))

if __name__=="__main__":
    start=time.time()
    return_status = main(2, write_file_dir, sids_rest[2:4])
    print('finished')
    print(f'status: {return_status}')
    print(f'Time (min): {(time.time()-start)/60:.3f}')
    print(dt.now())

# For n images, how many images m can be missing
# in order to compute the slice distance: 
# m<floor(n)/2