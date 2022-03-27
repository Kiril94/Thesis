""" 
Created on march 8th 2022
autor: Neus Rodeja Ferrer
"""

import pandas as pd 
from pathlib import Path
import nibabel as nib
import numpy as np
import os 

def load_nifti_img(filepath, dtype=None):
    '''
    NIFTI Image Loader
    :param filepath: path to the input NIFTI image
    :param dtype: dataio type of the nifti numpy array
    :return: return numpy array
    '''
    nim = nib.load(filepath)
    out_nii_array = np.array(nim.get_data(),dtype=dtype)
    out_nii_array = np.squeeze(out_nii_array) # drop singleton dim in case temporal dim exists
    meta = {'affine': nim.get_affine(),
            'dim': nim.header['dim'],
            'pixdim': nim.header['pixdim'],
            'name': os.path.basename(filepath)
            }

    return out_nii_array, meta 

data_path = Path("/home/cobra/data/Synth_CMB_sliced")
info_path = data_path / "rCMB_insCMB_scans_Info.csv"
all_info_path = data_path / "all_info_splitted.csv"
log_file = open(data_path/"log_masks_fix2.txt",'w')


df_info = pd.read_csv(info_path)
df_all_info = pd.read_csv(all_info_path)
df_merged = df_info.merge(df_all_info,how='left',left_on=['NIFTI File Name','z_position'], right_on=['NIFTI File Name','z_position'], suffixes=('_new','_old'))

#take only rCMB slices
df_rCMB = df_merged.drop_duplicates(['NIFTI File Name','z_position','x_position_new','y_position_new','set'])
df_slices = df_rCMB.groupby(['NIFTI File Name','z_position'])

for idx,slice in df_slices:

    file_name = slice['NIFTI File Name'].values[0]
    z = slice['z_position'].values[0]
    img_name = f'{file_name[:-7]}_slice{z}.nii.gz'

    mask_path = data_path / slice['set'].values[0] / "mask" / img_name
    # read mask 
    msk_old,_ = load_nifti_img(mask_path)
    msk_new = msk_old.copy()

    # add rCMB to masks 
    n_cmb_added = len(slice)
    for idx,row in slice.iterrows():
        x,y = row['x_position_new'],row['y_position_new']
        msk_new[ (x-1):(x+2),(y-1):(y+2)] = 1 

    #save mask again 
    nib.save(nib.Nifti1Image(msk_new,np.eye(4)),mask_path)

    log_file.write(f'{n_cmb_added},{img_name}\n')
    print(img_name)

log_file.close()