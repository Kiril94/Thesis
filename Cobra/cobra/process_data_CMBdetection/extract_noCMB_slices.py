# -*- coding: utf-8 -*-
"""
Created on Thu Dec 2 12:06:00 2021

@author: neusRodeja

Pipeline to extract the adjacent average 2D-slices from niftii images 
"""
import sys

sys.path.insert(0, '/home/neus/Documents/09.UCPH/MasterThesis/github/Thesis/Cobra/cobra/')

from access_sif_data.load_data_tools import load_nifti_img
import typer
import numpy as np
import nibabel as nib
import os
from rich.progress import track
from pathlib import Path
from typing import List

# command
# python create_masks.py "/media/neus/USB DISK/Synthetic_Cerebral_Microbleed_on_SWI_images/PublicDataShare_2020/rCMB_DefiniteSubject" "/media/neus/USB DISK/Synthetic_Cerebral_Microbleed_on_SWI_images/PublicDataShare_2020/sCMB_DefiniteSubject" "/media/neus/USB DISK/Synthetic_Cerebral_Microbleed_on_SWI_images/PublicDataShare_2020/sCMB_NoCMBSubject" -of "/media/neus/USB DISK/Synthetic_Cerebral_Microbleed_on_SWI_images/PublicDataShare_2020/processed" -lf ../tables/SynthCMB/all_info.csv
app = typer.Typer()

@app.command()
def slice_niftis(input_folder: List[Path],
                output_folder: Path = typer.Option(...,'--output-folder','-of',exists=True,help='Folder path where to save the sliced images.'),
):
    """Slice niftii."""
    
    file_names = []
    file_paths = []
    for folder in input_folder:
        dir_path,dir_name,fl_names = next(os.walk(folder))

        file_names = [*file_names,*fl_names]
        paths = [f'{dir_path}/{i}' for i in fl_names]
        file_paths = [*file_paths,*paths]
        
    if (not os.path.exists(f'{output_folder}/slices/images/')): os.makedirs(f'{output_folder}/slices/images/')
    if (not os.path.exists(f'{output_folder}/slices/masks/')): os.makedirs(f'{output_folder}/slices/masks/')
    print(len(file_names),len(file_paths))
    
    for i in track(range(len(file_paths))):
        file_path = file_paths[i]
        file_name = file_names[i]
       
        #read image
        img,_ = load_nifti_img(file_path)

        for idx_slice in range(img.shape[2]):                
            nib.save(nib.Nifti1Image(img[:,:,idx_slice],np.eye(4)),
            f"{output_folder}/slices/images/{file_name[:-7]}_slice{idx_slice}.nii.gz")
            nib.save(nib.Nifti1Image(np.zeros_like(img[:,:,idx_slice]),np.eye(4)),
            f"{output_folder}/slices/masks/{file_name[:-7]}_slice{idx_slice}.nii.gz")




if __name__ == "__main__":
    app()
