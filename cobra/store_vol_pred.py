import os
from os.path import join, split
from pathlib import Path
import pickle
import json
import pandas as pd
import numpy as np
from utilities.basic import list_subdir
import nibabel as nib
import time
from functools import partial
import multiprocessing as mp


script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parent

disk_dir = "F:"
disk_data_dir = join(disk_dir, 'CoBra', 'Data')
dcm_data_dir = join(disk_data_dir,'dcm')
data_dir = join(base_dir, 'data')
tables_dir = join(data_dir, 'tables')
data_long_dir = join(data_dir, 't1_longitudinal')
pred_dir = "F:\\CoBra\\Data\\volume_longitudinal_nii\\prediction"
brain_regions_df = pd.read_csv("F:\\CoBra\\Data\\volume_longitudinal_nii\\Brain_Regions.csv")
brain_regions_dic = pd.Series(brain_regions_df.Intensity.values,index=brain_regions_df.Region).to_dict()
with open(join(tables_dir, 'newIDs_dic.pkl'), 'rb') as f:
    id_dic = pickle.load(f)
with open(join(data_long_dir, "sids_long_new.pkl"), 'rb') as f:
    sids_ls = pickle.load(f)
with open(join(tables_dir, "disk_series_directories.json"), 'rb') as f:
    dir_dic = json.load(f)
inv_id_map = {v: k for k, v in id_dic.items()}

pred_files = list_subdir(pred_dir, ending='1mm_seg.nii.gz')
# we will create a list of dicts in parallel

def create_vol_dic(nii_file, brain_regions_dic, inv_id_map):
    arr = nib.load(nii_file).get_fdata()
    newid = split(nii_file)[1][:6]
    sid = inv_id_map[newid]
    volume_dic = {'newID':newid, 'SeriesInstanceUID':sid}

    for region, intensity in brain_regions_dic.items():
        volume_dic[region] = np.sum(arr==intensity)
    return volume_dic

def main(pred_files, brain_regions_dic, inv_id_map):
    print('Converting ', len(pred_files), 'files')
    create_vol_dic_part = partial(create_vol_dic, 
                    brain_regions_dic=brain_regions_dic,
                    inv_id_map=inv_id_map)
    start = time.time()
    with mp.Pool() as pool:
        volume_dic_ls = pool.map(create_vol_dic_part, pred_files)
    print(f'The conversion took {time.time()-start:.2d} s')
    print('Create df')
    df = pd.DataFrame(volume_dic_ls)
    return df


if __name__ == '__main__':
    df = main(pred_files[:5], brain_regions_dic, inv_id_map)
    print(df.head())