# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 10:58:38 2021
@author: klein
"""
#%% 
# In[Import]
import os, sys
from os.path import join, split
from pathlib import Path
import pandas as pd
from utilities import download
import pickle
import argparse


#%% 
# In[tables directories]

def main():#
    script_dir = os.path.realpath(__file__)
    base_dir = Path(script_dir).parent
    disk_dir = "F:"
    dst_data_dir = f"{disk_dir}/CoBra/Data/dcm"
    data_dir = join(base_dir, 'data')
    table_dir = join(data_dir, 'tables')
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch", required=True, type=int,
        help="batch to download, starting with 0")
    parser.add_argument("-bs", "--batch_size", required=False,default=1000, type=int,
        help="batch size")        
    parser.add_argument("-s", "--sids", required=True, type=str,
        help="File that contains sids to download")
    parser.add_argument("-u", "--update_d_files",required=False, default=False, type=str,
        help="Update downloaded files")
    args = parser.parse_args()
    update_downloaded_files = args.update_d_files
    batch = args.batch
    batch_size = args.batch_size
    sids_file_path = args.sids
    sids_file_name = (split(sids_file_path)[1]).split('.')[0]

    print("Load dataframes")
    df_volume_dir = pd.read_csv(join(table_dir, 'series_directories.csv'))
    df_patient_dir = pd.read_csv(join(table_dir, 'patient_directories.csv'))
    df_all = pd.read_csv(join(table_dir, 'scan_tables',"scan_after_sq_pred.csv"))
    print("Load dataframes finished")
    print("Using sids from the file: ", sids_file_path )
    with open(sids_file_path, 'rb') as f:
        group_list = pickle.load(f)
    if update_downloaded_files:
        print("Save list of already downloaded volumes")
        num_pat, num_vol = download.save_list_downloaded_volumes_and_patients()
        print(num_pat, "Patients and", num_vol, "volumes already downloaded")

    downloaded_ls = download.get_downloaded_volumes_ls()
    print('interesection', len(set(group_list).intersection(set(downloaded_ls))))
    group_list = list(set(group_list).difference(set(downloaded_ls)))
    print("Volumes still to download: ", len(group_list))
    use_batches = True
    # 5 batches are needed
    if use_batches:
        print("batch:", batch)
        start = 0
        df_group = df_all[df_all.SeriesInstanceUID.isin(group_list[start+batch*batch_size:start+batch_size*(batch+1)])]
    else:
        df_group = df_all[df_all.SeriesInstanceUID.isin(group_list)]
    df_group = df_group.sort_values('PatientID')                            

    print("Move ", len(df_group), "Volumes")
    print("Move ", df_group.PatientID.nunique(), "Patients")


    if use_batches:
        patient_log_file = join(base_dir, 'logs', f"{sids_file_name}_patient_log_{batch}.txt" )
        volume_log_file = join(base_dir, 'logs', f"{sids_file_name}_volume_log_{batch}.txt" )
    else:
        patient_log_file = join(base_dir, 'logs', f"{sids_file_name}_patient_log.txt" )
        volume_log_file = join(base_dir, 'logs', f"{sids_file_name}_volume_log.txt" )
    download.move_files_from_sif(df_group, df_volume_dir, df_patient_dir, 
                            dst_data_dir, patient_log_file, volume_log_file, src_dir="Y:\\")
if __name__ == '__main__':
    main()