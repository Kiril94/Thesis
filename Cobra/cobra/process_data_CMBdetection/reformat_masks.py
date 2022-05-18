"""
created on 21st april 2022
author: Neus Rodeja Ferrer

Reformat masks to have unique values 0 and 1 instead of 0 and 2**15
"""

import nibabel as nib
import sys
import os 
import numpy as np
from pathlib import Path

in_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])

print("IN_PATH",in_path)
print("OUT_PATH",out_path)

#save volume and mask with affine I
nib.openers.HAVE_INDEXED_GZIP=False
def load_nifti_img(filepath, dtype=None):
    '''
    NIFTI Image Loader
    :param filepath: path to the input NIFTI image
    :param dtype: dataio type of the nifti numpy array
    :return: return numpy array
    '''
    nim = nib.load(filepath)
    out_nii_array = np.array(nim.get_fdata(),dtype=dtype)
    out_nii_array = np.squeeze(out_nii_array) # drop singleton dim in case temporal dim exists
    meta = {'affine': nim.affine,
            'dim': nim.header['dim'],
            'pixdim': nim.header['pixdim'],
            'name': os.path.basename(filepath)
            }

    return out_nii_array, meta

#save no affines
files = next(os.walk(str(in_path)))[2]

for file in files:
    in_file_path = in_path / file
    out_file_path = out_path / file

    img,meta = load_nifti_img(in_file_path)

    img = img/np.max(img)
    img = img.astype(int)
    
    nib.save(nib.Nifti1Image(img,meta['affine']),out_file_path)