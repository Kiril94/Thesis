"""
created on march 1st 2022
autor: NEus Rodeja Ferrer
"""
import pandas as pd
import os
import shutil
from pathlib import Path

input_table =  "/home/lkw918/cobra/data/Synth_CMB_sliced/all_info_splitted.csv"
#input_table = Path(__file__).parent.parent / "tables" / "SynthCMB" / "all_info_splitted.csv"

data_path = "/home/lkw918/cobra/data/Synth_CMB_sliced"
df = pd.read_csv(input_table)

#create directories 
if (not os.path.exists(f'{data_path}/train/images')): os.makedirs(f'{data_path}/train/images')
if (not os.path.exists(f'{data_path}/train/masks')): os.makedirs(f'{data_path}/train/masks')

if (not os.path.exists(f'{data_path}/test/images')): os.makedirs(f'{data_path}/test/images')
if (not os.path.exists(f'{data_path}/test/masks')): os.makedirs(f'{data_path}/test/masks')

if (not os.path.exists(f'{data_path}/val/images')): os.makedirs(f'{data_path}/val/images')
if (not os.path.exists(f'{data_path}/val/masks')): os.makedirs(f'{data_path}/val/masks')

slices = df.groupby(['NIFTI File Name','z_position'])

n_total = len(slices)
n = 0
for idx,slice in slices:

    file_name = slice['NIFTI File Name'].values[0]
    z = slice['z_position'].values[0]
    img_name = f'{file_name[:-7]}_slice{z}.nii.gz'

    set = slice['set'].values[0]
    shutil.move(f'{data_path}/images/{img_name}',f'{data_path}/{set}/images/{img_name}')
    shutil.move(f'{data_path}/masks/{img_name}',f'{data_path}/{set}/masks/{img_name}')

    n += 1
    print(f'{n}/{n_total}')

