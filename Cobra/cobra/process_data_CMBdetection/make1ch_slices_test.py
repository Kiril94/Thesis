"""
created on april 4th 2022
autor: Neus Rodeja Ferrer
"""

import os
import nibabel as nib
import numpy as np

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


main_path = "/home/cobra/data/Synth_CMB_sliced_new/test"
folders = next(os.walk(main_path))[1]

for folder_view in next(os.walk(f"{main_path}"))[1]:

    print(folder_view)
    files = next(os.walk(f"{main_path}/{folder_set}/{folder_view}/images/"))[2]

    for file in files:
        img,meta = load_nifti_img(f"{main_path}/{folder_set}/{folder_view}/images/{}")
        nib.save(nib.Nifti1Image(img[:,:,1],meta),f"{main_path}/{folder_set}/{folder_view}/images_1ch/{file}")
