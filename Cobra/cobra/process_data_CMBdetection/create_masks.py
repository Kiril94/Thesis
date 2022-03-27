# -*- coding: utf-8 -*-
"""

NOT CORRECT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 

Created on Thu Dec 2 12:06:00 2021

@author: neusRodeja

Pipeline to extract the adjacent average 2D-slices from niftii images 
"""
import sys

sys.path.insert(0, '/home/neus/Documents/09.UCPH/MasterThesis/github/Thesis/Cobra/cobra/')

from access_sif_data.load_data_tools import load_nifti_img
import typer
from typing import Optional
from glob import iglob
import numpy as np
import pandas as pd
import nibabel as nib
import imageio
import os
from rich.progress import track
from enum import Enum
from pathlib import Path
from typing import List

# command
# python create_masks.py "/media/neus/USB DISK/Synthetic_Cerebral_Microbleed_on_SWI_images/PublicDataShare_2020/rCMB_DefiniteSubject" "/media/neus/USB DISK/Synthetic_Cerebral_Microbleed_on_SWI_images/PublicDataShare_2020/sCMB_DefiniteSubject" "/media/neus/USB DISK/Synthetic_Cerebral_Microbleed_on_SWI_images/PublicDataShare_2020/sCMB_NoCMBSubject" -of "/media/neus/USB DISK/Synthetic_Cerebral_Microbleed_on_SWI_images/PublicDataShare_2020/processed" -lf ../tables/SynthCMB/all_info.csv
app = typer.Typer()

class fileExtension(str,Enum):
    nifti = 'nii.gz'
    jpg = 'jpg'
    jpeg = 'jpeg'
    png = 'png'

@app.command()
def create_masks(input_folder: List[Path],
                output_folder: Path = typer.Option(...,'--output-folder','-of',exists=True,help='Folder path where to save the sliced images.'),
                labels_file: Path = typer.Option(...,'--labels-file','-lf',exists=True,help='csv file with image labels'),
                test_size: float = typer.Option(0.2,'--test-size','-ts',help='Size of the test set.'),
                val_size: float = typer.Option(0.1,'--val-size','-vs',help='Size of the validation set.'),
                input_file_extension: Optional[fileExtension]=typer.Option('nii.gz','--input-extension','-ie', help='Extension of the input files. No other formats implemented yet.'),
                output_file_extension: Optional[fileExtension]=typer.Option('nii.gz','--output-extension','-oe', help='Extension of the output files. No other formats implemented yet.'),
):
    """Extract the 2d slices that contain CMB and generate segmentation masks."""

    #read labels
    labels_df = pd.read_csv(labels_file)

    #split val,train,test
    val_df = labels_df.sample(frac=val_size)
    trainandtest_df = labels_df.drop(val_df.index)

    trainandtest_size = 1 - val_size    
    test_size = test_size / trainandtest_size
    test_df = trainandtest_df.sample(frac=test_size)
    train_df = trainandtest_df.drop(test_df.index)

    #config test,train
    datasets = {'val': val_df,
                'test': test_df,
                'train' : train_df }


    for set in datasets.keys():

        #make directories for masks and slices
        os.system(f'mkdir "{output_folder}/{set}"')
        os.system(f'mkdir "{output_folder}/{set}/images"')
        os.system(f'mkdir "{output_folder}/{set}/masks"')

        df = datasets[set]
        for i in track(range(df.shape[0]),description=f'Slicing {set} set... [{set} size is {df.shape[0]}]'):
            
            row = df.iloc[i]

            file_name = row['NIFTI File Name']
            x,y,z = row['x_position'],row['y_position'],row['z_position']

            #find file path
            for folder in input_folder:
                file_path = f"{folder}/{file_name}"
                if (os.path.exists(file_path)):
                    break
        
            #read image
            if (input_file_extension=='nii.gz'):
                img,_ = load_nifti_img(file_path)
            else:
                img = imageio.imread(file_path)

            #take slice 
            img_slice = img[:,:,z]
            img_mask = np.zeros_like(img_slice)
            img_mask[ (x-1):(x+2),(y-1):(y+2)] = 1 

            #save slice and mask 
            if (output_file_extension=='nii.gz'):
                img_name = file_name[:-7]
                nib.save(nib.Nifti1Image(img_slice,np.eye(4)),f"{output_folder}/{set}/images/{img_name}_slice{z}.{output_file_extension}")
                nib.save(nib.Nifti1Image(img_mask,np.eye(4)),f"{output_folder}/{set}/masks/{img_name}_slice{z}.{output_file_extension}")

            else:
                imageio.imwrite(f"{output_folder}/{set}/images/{file_name}_slice{z}.{output_file_extension}",img_slice.astype(np.uint8))
                imageio.imwrite(f"{output_folder}/{set}/masks/{file_name}_slice{z}.{output_file_extension}",img_mask.astype(np.uint8))



if __name__ == "__main__":
    app()
