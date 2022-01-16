from os.path import join
import os
import pandas as pd
from utilities import utils
from utilities.basic import list_subdir
from stats_tools import vis as svis
import numpy as np
import matplotlib.pyplot as plt
from pydicom import dcmread
import multiprocessing
import pickle
import json
import time
import sys

base_dir = "D:/Thesis/Cobra/cobra/"
sif_dir = 'Y:/'
fig_dir = join(base_dir, 'figs')
table_dir = join(base_dir, 'data/tables')

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
def get_distances_for_series(series_dir):
    dcm_dirs = list_subdir(series_dir)
    sys.stdout.flush()
    try:
        print('Use distance between slices')
        locations = []
        for dcm_dir in dcm_dirs[:3]:
            location = get_value_from_header(dcm_dir, 'SliceLocation')
            locations.append(location) 
        locations = np.array(locations)
        loc_diff = np.ediff1d(np.sort(locations))
        slice_dist = np.median(loc_diff)
        return slice_dist
    except:
        try:
            print('Use image orientation')
            cosines = get_value_from_header(dcm_dirs[0], 'ImageOrientationPatient')
            normal_indices = np.array([[1,2,0], [5,3,4], [2,0,1], [4,5,3]])
            normal = cosines[normal_indices[0]]*cosines[normal_indices[1]] - cosines[normal_indices[2]]*cosines[normal_indices[3]]
            distances = []
            for dcm_dir in dcm_dirs[:3]:
                ipp = get_value_from_header(dcm_dir, 'ImagePositionPatient')
                dist = np.sum(normal*ipp)
                distances.append(dist)
            dist_arr = np.array(distances)
            dist_arr_sorted = np.sort(dist_arr)
            dist_between_slices = np.ediff1d(dist_arr_sorted)
            return np.median(dist_between_slices) 
        except: return None

def save_distance_between_slices(sids, 
    dicoms_base_dir='Y:/'):
    """Save missing tags to text."""
    with open(join(base_dir, 'data/t1_longitudinal/sif_dir.json'), 'r') as fp:
        sif_dir_dic = json.load(fp)
    series_dirs = [join(dicoms_base_dir, sif_dir_dic[sid]) for sid in sids]
    
    with multiprocessing.Pool(4) as pool:
                distances = pool.map(get_distances_for_series, series_dirs[5:6])
    return distances


def main():
    with open(join(base_dir, 'data/t1_longitudinal/sim_3dt1_sids.json'), 'rb') as f:
        sids_3dt1 = json.load(f)
    distances = save_distance_between_slices(sids_3dt1)
    return distances
    
if __name__=="__main__":
    start=time.time()
    slice_distances = main()
    print(slice_distances)
    print('finished')
    print(time.time()-start)