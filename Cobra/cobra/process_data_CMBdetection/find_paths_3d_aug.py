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
log_file_path = main_path / "nonaug_aug_2d_3views_3dcnn_paths.csv"


#open log file
if (os.path.exists(log_file_path)):
    log_file = open(log_file_path,'a')
else:
    log_file = open(log_file_path,'w')
    log_file.write("input_file,output_file,input_mask,output_mask\n")


main_3dcnn_path = "/home/lkw918/cobra/data/cmb-3dcnn-data"
main_augmented_path = main_path / "train_aug"

## cmb data
folders_in_slices = next(os.walk(f"{main_3dcnn_path}/slices"))[1]

print(folders_in_slices)
for folder in folders_in_slices:
    cmb_slices = next(os.walk(f"{main_3dcnn_path}/slices/{folder}/images"))[2]
    print(folder)
    print(len(cmb_slices))
    for slice in cmb_slices:
        input_path = f"{main_3dcnn_path}/slices/{folder}/images/{slice}"
        input_mask_path = f"{main_3dcnn_path}/slices/{folder}/masks/{slice}"

        output_path = f"{main_augmented_path}/{folder}/images/{slice}"
        output_mask_path = f"{main_augmented_path}/{folder}/masks/{slice}"

        log_file.write(f"{input_path},{output_path},{input_mask_path},{output_mask_path}\n")

log_file.close()
