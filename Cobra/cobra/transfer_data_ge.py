# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 15:29:02 2021

@author: klein
"""

print('import')
import shutil
import os
from os.path import join, split, exists
from pathlib import Path
import time
import numpy as np
import pandas as pd
import sys
from datetime import datetime

cobra_dir = "D:/Thesis/Cobra/cobra"
if cobra_dir not in sys.path:
    sys.path.append(cobra_dir)
from utilities import stats
from multiprocessing import Pool


#define directories
print('define directories')
script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parent
src_dirs = os.listdir("Y:")
src_neg_dirs = sorted([f"{src_dirs}/{x}" for x \
                       in src_dirs if x.startswith('2019')])
dst_data_dir = "G:/CoBra/Data"
download_pat_path = join(base_dir, "data/share/Cerebriu/download_patients")
print('load dataframes')
volume_dir_df = pd.read_csv(join(base_dir, 'data/tables', 'sid_directories.csv'))
volume_dir_dic = pd.Series(
    volume_dir_df.Directory.values, index=volume_dir_df.SeriesInstanceUID).to_dict()
df_all = pd.read_csv(join(base_dir, "data/tables/neg_pos.csv"))
print('get batches')
dwi_t2s_gre = np.loadtxt(join(download_pat_path, "ge_dwi_t2s_gre.txt"),
                                   dtype='str')
gdtg = df_all[df_all['PatientID'].isin(dwi_t2s_gre)]
gdtg = gdtg.sort_values('PatientID')
patient_list_gdtg = gdtg.PatientID.unique()
batches = []
for i in range(6):
    if i<5:
        batches.append(patient_list_gdtg[i*50:(i+1)*50])
    else:
        batches.append(patient_list_gdtg[5*50:])
print('start download')
ge_dir = os.path.normpath("G:\CoBra\Data\GE")


def copy_batch(args): 
    """Takes tuple as arument with batch (list of patients) and batch_dir,
    with is the destination directory."""
    # printing process id to SHOW that we're actually using MULTIPROCESSING 
    batch, batch_dir = args[0], args[1]
    for pat in batch:
        start = time.time()
        patient_dir = volume_dir_df[
            stats.check_tags(
                volume_dir_df, tags=[pat], key='Directory')].iloc[0,1]
        patient_dir = os.path.normpath(patient_dir)
        patient_dir = join(*list(patient_dir.split(os.sep)[:2]))
        patient_dir = f"Y:\{patient_dir}"
        _, patient_id = split(patient_dir)
        dst_dir = join(batch_dir, patient_id)
        if exists(dst_dir):
            print(f"{dst_dir} exists" )
            continue
        print(f"{patient_dir}\n->{dst_dir}")
        now = datetime.now().time()
        print("now =", now)
        shutil.copytree(patient_dir, dst_dir)
        stop = time.time()
        print(f" {(stop-start)/60} mins")       
    print(f'parent process:{os.getppid()}')

if __name__=='__main__':
    first_batch = 1
    arg_list = [(batches[i], join(ge_dir, f"batch_{i}")) for \
                i in range(first_batch, 4)]
    with Pool(3) as p:
        p.map(copy_batch, arg_list)
    