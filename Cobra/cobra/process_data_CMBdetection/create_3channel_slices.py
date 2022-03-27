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

data_path = Path("/home/cobra/data")
sliced_folder = data_path / "Synth_CMB_sliced"
volumes_folder = data_path / "Synthetic_Cerebral_Microbleed_on_SWI_images"/"PublicDataShare_2020"
#volumes_folder = "/media/neus/USB DISK/Synthetic_Cerebral_Microbleed_on_SWI_images/PublicDataShare_2020"
all_info_path = sliced_folder / "all_info_splitted_v2.csv"
#all_info_path = Path(__file__).parent.parent /"tables"/"SynthCMB"/"all_info_splitted_v2.csv"

df_info = pd.read_csv(all_info_path)
df_info['group'] = df_info['NIFTI File Name'].str.split('_').map(lambda x: x[7] if len(x)==9 else 'rCMB' )

# select ttest slices
df_info = df_info[ df_info['NIFTI File Name'].isin(['242_T0_MRI_SWI_BFC_50mm_HM_sCMB_V1.nii.gz','	242_T1_MRI_SWI_BFC_50mm_HM_rsCMB_V1.nii.gz','242_T1_MRI_SWI_BFC_50mm_HM.nii.gz'])]
df_slices = df_info.groupby(['NIFTI File Name','z_position'])

dict_group_folders = {'sCMB':'sCMB_NoCMBSubject',
                    'rCMB':'rCMB_DefiniteSubject',
                    'rsCMB':'sCMB_DefiniteSubject',
                    }

for idx,slice in df_slices:

    file_name = slice['NIFTI File Name'].values[0]
    z = slice['z_position'].values[0]
    group_folder = dict_group_folders[slice['group'].values[0]]
    img_name = f'{file_name[:-7]}_slice{z}.nii.gz'

    #set paths
    slice_path = Path(__file__).parent / img_name #data_path / slice['set'].values[0] / "images_3channel" / img_name
    img_input_path = Path(volumes_folder) / group_folder / file_name
    
    # read volume 
    volume_img,_ = load_nifti_img(img_input_path)

    #select 3d channel slice
    slice_img = volume_img[:,:,(z-1):(z+2)]

    #save slice
    nib.save(nib.Nifti1Image(slice_img,np.eye(4)),slice_path)

    print(img_name)
