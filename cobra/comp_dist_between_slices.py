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
base_dir = "D:/Thesis/Cobra/cobra/"
sif_dir = 'Y:/'
fig_dir = join(base_dir, 'figs')
table_dir = join(base_dir, 'data/tables')

def get_slice_location(dcm_dir):
    dcm = dcmread(dcm_dir)
    return dcm['SliceLocation']

def save_distance_between_slices(sids, 
    dicoms_base_dir='Y:/'):
    """Save missing tags to text."""
    with open(join(base_dir, 'data/t1_longitudinal/sif_dir.json'), 'r') as fp:
        sif_dir_dic = json.load(fp)
    for i, sid in enumerate(sids[:1]):
        series_dir = join(dicoms_base_dir, sif_dir_dic[sid])
        dicom_dirs = list_subdir(series_dir)
        with multiprocessing.Pool(2) as pool:
            locations = pool.map(get_slice_location, dicom_dirs)
        return locations


def main():
    with open(join(base_dir, 'data/t1_longitudinal/sim_3dt1_sids.json'), 'rb') as f:
        sids_3dt1 = json.load(f)
    locations = save_distance_between_slices(sids_3dt1)
    print(locations)
    
if __name__=="__main__":
    start=time.time()
    main()
    print('finished')
    print(time.time()-start)