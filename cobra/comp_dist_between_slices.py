from os.path import join, split, normpath
import os
from utilities.basic import list_subdir
import numpy as np
from pydicom import dcmread
import multiprocessing
import json
import time
import sys
from pathlib import Path


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


def get_distances_for_series(series_dir, aggregating_func=np.min):
    try:
        dcm_dirs = list_subdir(series_dir)
    except Exception as e:
        print("ERROR : "+str(e))
        return None
    sys.stdout.flush()
    try:
        return compute_dist_from_slice_location(dcm_dirs, aggregating_func)
    except:
        try:
            return compute_dist_from_img_pos_and_orientation(dcm_dirs, aggregating_func) 
        except: return None

def save_distance_between_slices(sids, 
    dicoms_base_dir='F:/'):
    """Save missing tags to text."""
    with open(join(base_dir, 'data/t1_longitudinal/sif_dir.json'), 'r') as fp:
        sif_dir_dic = json.load(fp)
    patient_dirs = [join(*normpath(sif_dir_dic[sid]).split(os.sep)[:2]) for sid in sids]
    print(patient_dirs[0])
    series_dirs = [join(dicoms_base_dir, patient_dirs[i], sid ) \
                for i, sid in enumerate(sids)]
    
    with multiprocessing.Pool(2) as pool:
                distances = pool.map(get_distances_for_series, 
                        series_dirs)
    return distances




script_dir = os.path.realpath(__file__)
base_dir = join(Path(script_dir).parent, 'cobra')
dicom_base_dir = "F:/CoBra/Data/dcm"
def main():
    with open(join(base_dir, 'data/t1_longitudinal/sim_3dt1_sids.json'), 'rb') as f:
        sids_3dt1 = json.load(f)
    distances = save_distance_between_slices(sids_3dt1[-10:], 
            dicoms_base_dir=dicom_base_dir )
    return distances
    
if __name__=="__main__":
    start=time.time()
    slice_distances = main()
    print(slice_distances)
    print('finished')
    print(time.time()-start)

# For n images, how many images m can be missing
# in order to compute the slice distance: 
# m<floor(n)/2