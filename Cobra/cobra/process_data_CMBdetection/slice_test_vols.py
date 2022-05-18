"""
created on march 30th 2022
author: Neus Rodeja Ferrer
"""

#%%
import pandas as pd 
import os 
import nibabel as nib
import numpy as np 
import time
import datetime

# FOR SYNTH CMB 
#cmb_info_path = "/home/neus/Documents/09.UCPH/MasterThesis/github/Thesis/Cobra/cobra/tables/SynthCMB/all_info_splitted_v2.csv"
cmb_info_path = "/home/lkw918/cobra/data/Synthetic_Cerebral_Microbleed_on_SWI_images/all_info_splitted_v2.csv"
cmb_nii_path = "/home/lkw918/cobra/data/Synthetic_Cerebral_Microbleed_on_SWI_images/PublicDataShare_2020" # "/media/neus/USB DISK/cmb-3dcnn-data/nii"
cmb_masks_path = "/home/lkw918/cobra/data/Synthetic_Cerebral_Microbleed_on_SWI_images/PublicDataShare_2020/3d_masks"
cmb_slices_path = "/home/lkw918/cobra/data/Synth_CMB_sliced_new"

#configuration for SYNTH
r_xy_patch, r_z_patch = 6,3
border_xy_patch, border_z_patch = 2,1
frac_cmb_thresh = 0.52

# #configuration for 3dCNN
# r_xy_patch, r_z_patch = 12,6
# border_xy_patch, border_z_patch = 4,2
# frac_cmb_thresh = 0.65

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

#%% #create masks

#read cmb info
df_cmb_info = pd.read_csv(cmb_info_path)

#filter on test set
df_cmb_info = df_cmb_info[ df_cmb_info['set']=='test' ]

# CONFIGURATION FOR SYNTH CMB
dict_group_folders = {'sCMB':'sCMB_NoCMBSubject',
                    'rCMB':'rCMB_DefiniteSubject',
                    'rsCMB':'sCMB_DefiniteSubject',
                    }
df_cmb_info['group'] = df_cmb_info['NIFTI File Name'].str.split('_').map(lambda x: x[7] if len(x)==9 else 'rCMB' )

df_cmb_info['group_folder'] = df_cmb_info['group'].map(lambda x: dict_group_folders[x])
#df_cmb_info['path'] = f"{cmb_nii_path}/{df_cmb_info['group_folder']}/{df_cmb_info['NIFTI File Name']}"
df_cmb_info.rename(columns = {'NIFTI File Name': 'file_name',
                                'x_position': 'x_pos',
                                'y_position': 'y_pos',
                                'z_position': 'z_pos'},inplace=True)

#create thresh function
apply_thresh = np.vectorize(lambda x: 1 if x>frac_cmb_thresh else 0)


#find volumes
vol_groups = df_cmb_info.groupby("file_name")

start_time = time.time()
for idx_vol,vol_info in vol_groups:
    filename = vol_info['file_name'].values[0]
    print(filename)
    #read nifti
    img_path = f"{cmb_nii_path}/{vol_info['group_folder'].values[0]}/{filename}" 
    msk_path = f"{cmb_masks_path}/{filename}"
    img,_ = load_nifti_img(img_path)
    mask,_ = load_nifti_img(msk_path)

    #save sagittal slices
    for x in range(img.shape[0]):
        #slice_img = img[x-1:x+2,:,:]
        slice_img = img[x,:,:]
        slice_img = np.moveaxis(slice_img,0,-1)


        slice_name = f'{filename[:-7]}_slice{x}.nii.gz'
        path_to_save = f"{cmb_slices_path}/{vol_info['set'].values[0]}/sagittal"
        nib.save(nib.Nifti1Image(slice_img,np.eye(4)),f"{path_to_save}/images_1ch/{slice_name}")

        if (x==0 or x==(img.shape[0]-1)):
            slice_msk = mask[x,:,:]
            nib.save(nib.Nifti1Image(slice_msk,np.eye(4)),f"{path_to_save}/masks/{slice_name}")


    #save coronal slices
    for y in range(img.shape[1]):
        #slice_img = img[:,y-1:y+2,:]
        slice_img = img[:,y,:]
        slice_img = np.moveaxis(slice_img,1,-1)
        slice_msk = mask[:,y,:]

        slice_name = f'{filename[:-7]}_slice{y}.nii.gz'
        path_to_save = f"{cmb_slices_path}/{vol_info['set'].values[0]}/coronal"
        nib.save(nib.Nifti1Image(slice_img,np.eye(4)),f"{path_to_save}/images_1ch/{slice_name}")

        if (y==0 or y==(img.shape[1]-1)):
            slice_msk = mask[:,y,:]
            nib.save(nib.Nifti1Image(slice_msk,np.eye(4)),f"{path_to_save}/masks/{slice_name}")

    #save axial slices
    for z in range(img.shape[2]):
        #slice_img = img[:,:,z-1:z+2]
        slice_img = img[:,:,z]
        slice_msk = mask[:,:,z]

        slice_name = f'{filename[:-7]}_slice{z}.nii.gz'
        path_to_save = f"{cmb_slices_path}/{vol_info['set'].values[0]}/axial"
        nib.save(nib.Nifti1Image(slice_img,np.eye(4)),f"{path_to_save}/images_1ch/{slice_name}")

        if (z==0 or z==(img.shape[2]-1)):
            slice_msk = mask[:,:,z]
            nib.save(nib.Nifti1Image(slice_msk,np.eye(4)),f"{path_to_save}/masks/{slice_name}")