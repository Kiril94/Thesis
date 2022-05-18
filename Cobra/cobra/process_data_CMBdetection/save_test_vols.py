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
import shutil

# FOR SYNTH CMB 
#cmb_info_path = "/home/neus/Documents/09.UCPH/MasterThesis/github/Thesis/Cobra/cobra/tables/SynthCMB/all_info_splitted_v2.csv"
cmb_info_path = "/home/lkw918/cobra/data/Synthetic_Cerebral_Microbleed_on_SWI_images/all_info_splitted_v2.csv"
cmb_nii_path = "/home/lkw918/cobra/data/Synthetic_Cerebral_Microbleed_on_SWI_images/PublicDataShare_2020" # "/media/neus/USB DISK/cmb-3dcnn-data/nii"
cmb_masks_path = "/home/lkw918/cobra/data/Synthetic_Cerebral_Microbleed_on_SWI_images/PublicDataShare_2020/3d_masks"
cmb_new_vols_path = "/home/lkw918/cobra/data/volumetric_data/test"

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

    img_path = f"{cmb_nii_path}/{vol_info['group_folder'].values[0]}/{filename}" 
    msk_path = f"{cmb_masks_path}/{filename}"

    img_dst_path = f"{cmb_new_vols_path}/images/"
    msk_dst_path = f"{cmb_new_vols_path}/masks/"

    shutil.copy(img_path,img_dst_path)
    shutil.copy(msk_path,msk_dst_path)



