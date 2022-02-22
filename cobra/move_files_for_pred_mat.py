#%%
import shutil
import sys
import os
from os.path import join, split
from pathlib import Path
import pandas as pd
import numpy as np
import gzip
import multiprocessing as mp
from dcm2nii import dcm2nii
from datetime import datetime as dt
import time
import json
import pickle
from utilities import basic, fix_dcm_incomplete_vols
from pydicom import dcmread
from utilities.basic import get_dir, make_dir, remove_file, get_part_of_path, get_proc_id
import matlab.engine
eng = matlab.engine.start_matlab()
eng.addpath(r'C:\\Users\\kiril\\Thesis\\CoBra\\cobra\dcm2nii\dcm2nii_mat\\functions', nargout=0)
eng.addpath(r'C:\\Users\\kiril\\Thesis\\CoBra\\cobra\dcm2nii\dcm2nii_mat\dcm2nii')
def get_missing_files(sids_to_download, download_dir, newid_dic):
    """
    sids_to_download: List of SeriesInstanceUIDs that need to be downloaded
    download_dir: str, directory where downloaded files are placed
    newid_dic: dictionary used to map sids to 6 digit new ids
    returns: list of missing files sids
    """
    downloaded_files_ids = [file[:-7] for file in os.listdir(download_dir)]
    downloaded_files_sids = [newid_dic[id] for id in downloaded_files_ids]
    missing_files = set(sids_to_download).difference(set(downloaded_files_sids))
    return list(missing_files)

def dcm2nii_mat():
    eng.dcm2nii_main('F:\\CoBra\\Data\\dcm\\2019_01\\00ade8f21e97e455352491aab6b00cb3', 'F:\\CoBra\\Data\\test')
if __name__ == '__main__':
    dcm2nii_mat()
    #print(eng.foo(1,0))