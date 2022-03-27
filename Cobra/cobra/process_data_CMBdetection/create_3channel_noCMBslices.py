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

data_path = Path("/home/lkw918/cobra/data")
sliced_folder = data_path / "NoCMB_sliced" / "slices" / "images_3channel"
volumes_folder = data_path / "Synthetic_Cerebral_Microbleed_on_SWI_images"/"PublicDataShare_2020" / "NoCMBSubject"

nocmb_volumes = next(os.walk(volumes_folder))[2]

for filename in nocmb_volumes:

    vol_path = Path(volumes_folder) / filename    
    # read volume 
    volume_img,_ = load_nifti_img(vol_path)

    for z in range(1,volume_img.shape[2]-1):        
        #select 3d channel slice
        slice_img = volume_img[:,:,(z-1):(z+2)]

        slice_name = f'{filename[:-7]}_slice{z}.nii.gz'
        slice_path = Path(sliced_folder) / slice_name 
        #save slice
        nib.save(nib.Nifti1Image(slice_img,np.eye(4)),slice_path)

    print(slice_name)

#%%

path_obj_cph = "/home/neus/Documents/09.UCPH/MasterThesis/DATA/test_pipelines_cph_data/objects_reshaped_positions.csv"