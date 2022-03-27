"""
created on 1st march 2022
autor: Neus Rodeja Ferrer
"""

import pandas as pd
from pathlib import Path
import numpy as np
import nibabel as nib

h,w = 176,256
input_table =  "/home/lkw918/cobra/data/Synth_CMB_sliced/all_info_splitted.csv"
#input_table = Path(__file__).parent.parent / "tables" / "SynthCMB" / "all_info_splitted.csv"
mask_path = "/home/lkw918/cobra/data/Synth_CMB_sliced/masks"
log_file = open("/home/lkw918/cobra/data/Synth_CMB_sliced/log_masks_fix.txt",'w')
df = pd.read_csv(input_table)

slices = df.groupby(['NIFTI File Name','z_position'])

for idx,slice in slices:

    if (len(slice)<2): continue

    file_name = slice['NIFTI File Name'].values[0]
    z = slice['z_position'].values[0]
    img_name = f'{file_name[:-7]}_slice{z}.nii.gz'

    img_mask = np.zeros(shape=(h,w))

    for idx,row in slice.iterrows():
        x,y = row['x_position'],row['y_position']
        img_mask[ (x-1):(x+2),(y-1):(y+2)] = 1 

    nib.save(nib.Nifti1Image(img_mask,np.eye(4)),f"{mask_path}/{img_name}")
    
    log_file.write(f'{len(slice)},{img_name}\n')
    print(img_name)

log_file.close()