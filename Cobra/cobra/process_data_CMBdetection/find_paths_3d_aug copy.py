"""
created on march 30th 2022
author: Neus Rodeja Ferrer
"""

from numpy import NaN
import pandas as pd 
from pathlib import Path
import nibabel as nib
import numpy as np
import os 
import time

#%% 

main_path = Path("/home/lkw918/cobra/data/Synth_CMB_sliced_new")
#main_path = Path("/home/neus/Documents/09.UCPH/MasterThesis/github/MultiResUNet_cmb/cmb_data/new_slices")
log_file_path = main_path / "nonaug_aug_2d_3views_3dcnn_paths.csv"

dict_orig_folder_to_dest_folders = {'train':'train_aug',
                                    'val':'val_aug'}

#open log file
if (os.path.exists(log_file_path)):
    log_file = open(log_file_path,'a')
else:
    log_file = open(log_file_path,'w')
    log_file.write("input_file,output_file,input_mask,output_mask\n")

# for orig_folder in dict_orig_folder_to_dest_folders.keys():

#     print(main_path/orig_folder/"images")
#     folders_in_folder = next(os.walk(main_path/orig_folder))[1] #axial,sagittal,coronal

#     for orig_subfolder in folders_in_folder:

#         files_in_folder = next(os.walk(main_path/orig_folder/orig_subfolder/"images"))[2]

#         for file in files_in_folder:
#             dst_folder = dict_orig_folder_to_dest_folders[orig_folder]

#             orig_path = main_path/orig_folder/orig_subfolder/"images"/file
#             dst_path = main_path/dst_folder/orig_subfolder/"images"/file

#             orig_mask_path = main_path/orig_folder/orig_subfolder/"masks"/file
#             dst_mask_path = main_path/dst_folder/orig_subfolder/"masks"/file

#             log_file.write(f"{orig_path},{dst_path},{orig_mask_path},{dst_mask_path}\n")

# log_file.close()

# %%
#tables_path = Path(__file__).parent.parent / "tables" 
#create file with paths for 3d data augmentation

main_synth_path = "/home/lkw918/cobra/data/Synthetic_Cerebral_Microbleed_on_SWI_images/PublicDataShare_2020" 
main_3dcnn_path = "/home/lkw918/cobra/data/cmb-3dcnn-data"

main_augmented_path = "/home/lkw918/cobra/data/augmented_volumes"

# log_file_path = tables_path / "nonaug_3d_aug_paths.csv"

# #start with Synthetic data
# all_info_synth = tables_path/"SynthCMB"/"all_info_splitted_v2.csv"
# info_synth = pd.read_csv(all_info_synth)

# #filter on test set
# info_synth = info_synth[ info_synth['set'].isin(['train','val']) ]

# # find input folders
# dict_group_folders = {'sCMB':'sCMB_NoCMBSubject',
#                     'rCMB':'rCMB_DefiniteSubject',
#                     'rsCMB':'sCMB_DefiniteSubject',
#                     }
# info_synth['group'] = info_synth['NIFTI File Name'].str.split('_').map(lambda x: x[7] if len(x)==9 else 'rCMB' )
# info_synth['group_folder'] = info_synth['group'].map(lambda x: dict_group_folders[x])

# input_path = f"{main_synth_path}/{info_synth['group_folder']}"
# #find volumes
# vol_groups = info_synth.groupby("NIFTI File Name")

# #open log file
# if (os.path.exists(log_file_path)):
#     log_file = open(log_file_path,'a')
# else:
#     log_file = open(log_file_path,'w')
#     log_file.write("input_file,output_file,input_mask,output_mask\n")

# start_time = time.time()
# for idx_vol,vol_info in vol_groups:
#     filename = vol_info['NIFTI File Name'].values[0]

#     input_path = f"{main_synth_path}/{vol_info['group_folder'].values[0]}/{filename}"
#     input_mask_path = f"{main_synth_path}/3d_masks/{filename}"

#     output_path = f"{main_augmented_path}/{vol_info['set'].values[0]}/images/{filename}"
#     output_mask_path = f"{main_augmented_path}/{vol_info['set'].values[0]}/masks/{filename}"

#     log_file.write(f"{input_path},{output_path},{input_mask_path},{output_mask_path}\n")

## cmb data
folders_in_slices = next(os.walk(f"{main_3dcnn_path}/slices"))[1]

for folder in folders_in_slices:
    cmb_slices = next(os.walk(f"{main_3dcnn_path}/slices/{folder}"))[2]

    for slice in cmb_slices:
        input_path = f"{main_3dcnn_path}/slices/{folder}/images/{slice}"
        input_mask_path = f"{main_3dcnn_path}/slices/{folder}/masks/{slice}"

        output_path = f"{main_augmented_path}/train/images/{vol}.gz"
        output_mask_path = f"{main_augmented_path}/train/masks/{vol}.gz"

        log_file.write(f"{input_path},{output_path},{input_mask_path},{output_mask_path}\n")

log_file.close()
