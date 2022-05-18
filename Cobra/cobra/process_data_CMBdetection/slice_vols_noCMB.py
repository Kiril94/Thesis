"""
created on march 29th 2022
author: Neus Rodeja Ferrer
"""

import nibabel as nib
import numpy as np
import os

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
    meta = {'affine': nim.affine, #nim.get_affine(),
            'dim': nim.header['dim'],
            'pixdim': nim.header['pixdim'],
            'name': os.path.basename(filepath)
            }

    return out_nii_array, meta 

vols_path = "/home/lkw918/cobra/data/Synthetic_Cerebral_Microbleed_on_SWI_images/PublicDataShare_2020/NoCMBSubject"

slices_path =  "/home/lkw918/cobra/data/NoCMB_sliced/slices_new"

filenames = next(os.walk(vols_path))[2]

for filename in filenames:

    img,_ = load_nifti_img(f"{vols_path}/{filename}")
    mask = np.zeros_like(img)

    #save sagittal slices
    for x in range(1,img.shape[0]-1):
        slice_img = img[x-1:x+2,:,:]
        slice_img = np.moveaxis(slice_img,0,-1)
        slice_msk = mask[x,:,:]

        slice_name = f'{filename[:-7]}_slice{x}.nii.gz'
        path_to_save = f"{slices_path}/sagittal"
        nib.save(nib.Nifti1Image(slice_img,np.eye(4)),f"{path_to_save}/images/{slice_name}")
        nib.save(nib.Nifti1Image(slice_msk,np.eye(4)),f"{path_to_save}/masks/{slice_name}")


    #save coronal slices
    for y in range(1,img.shape[1]-1):
        slice_img = img[:,y-1:y+2,:]
        slice_img = np.moveaxis(slice_img,1,-1)
        slice_msk = mask[:,y,:]

        slice_name = f'{filename[:-7]}_slice{y}.nii.gz'
        path_to_save = f"{slices_path}/coronal"
        nib.save(nib.Nifti1Image(slice_img,np.eye(4)),f"{path_to_save}/images/{slice_name}")
        nib.save(nib.Nifti1Image(slice_msk,np.eye(4)),f"{path_to_save}/masks/{slice_name}")

    #save axial slices
    for z in range(1,img.shape[2]-1):
        slice_img = img[:,:,z-1:z+2]
        slice_msk = mask[:,:,z]

        slice_name = f'{filename[:-7]}_slice{z}.nii.gz'
        path_to_save = f"{slices_path}/axial"
        nib.save(nib.Nifti1Image(slice_img,np.eye(4)),f"{path_to_save}/images/{slice_name}")
        nib.save(nib.Nifti1Image(slice_msk,np.eye(4)),f"{path_to_save}/masks/{slice_name}")