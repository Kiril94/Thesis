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
from utilities.basic import get_dir, list_subdir, make_dir, remove_file, get_part_of_path, get_proc_id
import matlab.engine

script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parent
tables_dir = join(base_dir, 'data', 'tables')
disk_data_dir = join("F:\\", 'CoBra', 'Data')
nii_input_dir = join(disk_data_dir, 'volume_longitudinal_nii', 'input', 'nii_files')


eng = matlab.engine.start_matlab()
eng.addpath('C:\\Users\\kiril\\Thesis\\CoBra\\cobra\\dcm2nii\\dcm2nii_mat\\functions', nargout=0)
eng.addpath('C:\\Users\\kiril\\Thesis\\CoBra\\cobra\\dcm2nii\\dcm2nii_mat\\dcm2nii')
eng.addpath('C:\\Users\\kiril\\Thesis\\CoBra\\cobra\\dcm2nii\\dcm2nii_mat\\spm12')


with open(join(tables_dir, 'newIDs_dic.pkl'), 'rb') as f:
    id_dic = pickle.load(f)

with open("C:\\Users\\kiril\\Thesis\\CoBra\\cobra\\data\\t1_longitudinal\\pairs_3dt1_long_sids.pkl", 'rb') as f:
    sids_ls = pickle.load(f)

with open("C:\\Users\\kiril\\Thesis\\CoBra\\cobra\\data\\tables\\disk_series_directories.json", 'rb') as f:
    dir_dic = json.load(f)


def get_missing_files(sids_to_download, download_dir, newid_dic):
    """
    sids_to_download: List of SeriesInstanceUIDs that need to be downloaded
    download_dir: str, directory where downloaded files are placed
    newid_dic: dictionary used to map sids to 6 digit new ids
    returns: list of missing files sids
    """
    inv_map = {v: k for k, v in newid_dic.items()}
    downloaded_files_ids = [file[:-7] for file in os.listdir(download_dir)]
    downloaded_files_sids = [inv_map[id] for id in downloaded_files_ids]
    missing_files = set(sids_to_download).difference(set(downloaded_files_sids))
    return list(missing_files)

def dcm2nii_mat_main(src_dir, tmp_dir, tgt_dir, tgt_name):
    """src_dir: """
    tmp_dir_sp = join(tmp_dir, get_proc_id())
    make_dir(tmp_dir_sp)
    try:
        eng.dcm2nii_main(src_dir, tmp_dir)
    except:
        print("dcm2nii failed, try spm")
        try:
            eng.spm12_main(src_dir, tmp_dir)
        except:
            for f in list_subdir(tmp_dir):
                shutil.remove(f)
            print('x')
    out_files = list_subdir(tmp_dir, ending='.nii.gz')
    assert len(out_files)==0, f'More than 1 nii file was created for {src_dir}'
    shutil.move(out_files[0], join(tgt_dir, tgt_name+'.nii.gz'))
    return 0

if __name__ == '__main__':
    #dcm2nii_mat_main()
    src_test = "F:\CoBra\Data\dcm\\2019_06\\00a4185a2d35d0fb20f16747fa8b9d36\\7c7bff9d964225ac568dea16f5b69987"
    tgt_test = "F:\\CoBra\\Data\\test"
    missing_files = get_missing_files(sids_ls, nii_input_dir, id_dic)
    print(len(missing_files))
    #print(missing_files[:2])
    print(len(sids_ls))
    for f in missing_files[:3]:
        sid = split(f)[1]
        new_id = id_dic[sid]
        src_dir = dir_dic[sid]
        dcm2nii_mat_main(src_dir, tgt_test, tgt_test, new_id)