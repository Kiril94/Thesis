"""Creating a csv file which contains all the predicted brain volumes.
Modifies existing df stored under 
join('t1_longitudinal', 'results', 'volume_prediction_results.feather').
Files which are already in this dataframe are skipped by default!! 
When reruning volume prediction for some files
either remove them from the dataframe (recommended) or set converted_files_df to None.
"""

import os
import sys
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
disk_dir = "G:"
disk_data_dir = join(disk_dir, 'CoBra', 'Data')
dcm_data_dir = join(disk_data_dir,'dcm')
data_dir = join(base_dir, 'data')
tables_dir = join(data_dir, 'tables')
data_long_dir = join(data_dir, 't1_longitudinal')
pred_dirs = [join(disk_data_dir,"volume_longitudinal_nii\\prediction-new"), 
            join(disk_data_dir,"volume_cross_nii\\prediction-new")] 
brain_regions_df = pd.read_csv(join(disk_data_dir, "volume_longitudinal_nii\\Brain_Regions.csv"))
brain_regions_dic = pd.Series(brain_regions_df.Intensity.values,
    index=brain_regions_df.Region).to_dict()
with open(join(tables_dir, 'newIDs_dic.pkl'), 'rb') as f:
    id_dic = pickle.load(f)
with open(join(data_long_dir, "sids_long_new.pkl"), 'rb') as f:
    sids_ls = pickle.load(f)
with open(join(tables_dir, "disk_series_directories.json"), 'rb') as f:
    dir_dic = json.load(f)
inv_id_map = {v: k for k, v in id_dic.items()}
pred_df0 = pd.read_csv(join(data_dir,"volume_pred", "volume_prediction_results0.csv"))
pred_df0.newID = pred_df0.newID.map(lambda x: str(x).zfill(6))
ids_present = []#[nid+'_1mm_seg.nii.gz' for nid in pred_df0.newID]
pred_files = []
for pred_dir in pred_dirs:
    pred_files = pred_files + list_subdir(pred_dir, ending='1mm_seg.nii.gz', exclude=ids_present)


def create_vol_dic(nii_file, brain_regions_dic, inv_id_map):
    sys.stdout.flush()
    print('.', end='')
    arr = nib.load(nii_file).get_fdata()
    newid = split(nii_file)[1][:6]
    sid = inv_id_map[newid]
    volume_dic = {'newID':newid, 'SeriesInstanceUID':sid}
    for region, intensity in brain_regions_dic.items():
        volume_dic[region] = np.sum(arr==intensity)
    return volume_dic

def main(pred_files, brain_regions_dic, inv_id_map, converted_files_df=None,
            batch_size=500, name="volume_prediction_results_new", test=False):
    sys.stdout.flush()
    print('Reading ', len(pred_files), 'files in batches of ', batch_size)

    if test:
        batch_size = 5
    for i in range(0, len(pred_files), batch_size):
        if test:
            if i>2*batch_size:
                break
        batch = pred_files[i:i+batch_size]
        if not isinstance(converted_files_df, type(None)):
            stored_ids = converted_files_df.newID.tolist()
            batch = [file for file in batch if split(file)[1][:6] not in stored_ids]
            sys.stdout.flush()
            print(len(batch), "after exclusion of converted files")
        
        create_vol_dic_part = partial(create_vol_dic, 
                        brain_regions_dic=brain_regions_dic,
                        inv_id_map=inv_id_map)
        start = time.time()
        with mp.Pool() as pool:
            volume_dic_ls = pool.map(create_vol_dic_part, batch)
        sys.stdout.flush()
        print(f'The storing took {(time.time()-start)/60:.2f} min')
        print('Create df')
        df = pd.DataFrame(volume_dic_ls)
        if not isinstance(converted_files_df, type(None)):
            df = pd.concat([converted_files_df, df], ignore_index=True)
        else:
            converted_files_df = df
        sys.stdout.flush()
        print("Save temporary dataframe")
        df.to_csv(join(data_dir, "volume_pred", name + "_temp.csv"), index=None)
    return df


if __name__ == '__main__':
    print(len(pred_files), 'segmented volumes to be computed')
    df = main(pred_files, brain_regions_dic, inv_id_map, converted_files_df=None, test=True)
    #df.to_feather(join(data_dir, "volume_pred","volume_prediction_results_new.feather"))
    #df.to_feather(join(disk_data_dir, "volume_pred_results", "volume_prediction_results_new.feather"))
    df.to_csv(join(data_dir, "volume_pred", "volume_prediction_results_new.csv"), index=None)
    #df.to_csv(join("Y:\\tables", "volume_prediction_results_new.csv"), index=None)