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
from typing import Optional
from glob import iglob
import numpy as np
import pandas as pd
import nibabel as nib
import imageio
import os
from shutil import copy
from rich.progress import track
from sklearn.model_selection import train_test_split
from enum import Enum
import xml.etree.ElementTree as ET

app = typer.Typer()

IMAGE_WIDTH = 256 #x direction, index 1 (cols)
IMAGE_HEIGHT = 176 #y direction, index 0 (rows)
# the image width  and height need to be multiple of 32
ROWS_TO_EXCLUDE_PER_SIDE = 8 
COLUMNS_TO_EXCLUDE_PER_SIDE = 0

BOUNDING_BOX_WIDTH = 7/IMAGE_WIDTH
BOUNDING_BOX_HEIGHT = 7/IMAGE_HEIGHT

BOUNDING_BOX_HALF_WIDTH = 3
BOUNDING_BOX_HALF_HEIGHT = 3

# python preprocess.py preprocess-yolo '/media/neus/USB DISK/Synthetic_Cerebral_Microbleed_on_SWI_images/PublicDataShare_2020/rCMB_DefiniteSubject' /home/neus/Documents/09.UCPH/MasterThesis/DATA/prova_training_YOLO /home/neus/Documents/09.UCPH/MasterThesis/github/Thesis/Cobra/cobra/tables/SynthCMB/rCMBInformationInfo.csv 0.25 --overwrite
class SliceContext(str,Enum):
    average = 'average'

class fileExtension(str,Enum):
    nifti = 'nii.gz'
    jpg = 'jpg'
    jpeg = 'jpeg'
    png = 'png'

@app.command()
def extract_1channel_slices(input_folder: str,
                            output_folder: str,
                            slice_context: Optional[SliceContext]=typer.Option('average','--slice-context','-sc', help='Mode to take the context of a slice.'),
                            input_file_extension: Optional[fileExtension]=typer.Option('nii.gz','--input-extension','-ie', help='Extension of the input files. No other formats implemented yet.'),
                            output_file_extension: Optional[fileExtension]=typer.Option('nii.gz','--output-extension','-oe', help='Extension of the output files. No other formats implemented yet.'),
):
    """Extract the averaged slices from 3d niftii files."""


    nifti_files = [file for file in iglob(f"{input_folder}/*.{input_file_extension}")]
    for file in track(nifti_files,description='Slicing files...'):
        img_name = file.split('/')[-1][:-len(input_file_extension)-1]
        img_array,_ = load_nifti_img(file)
        h,w,d = img_array.shape

        #adapt dimensions to be multiple of 32
        if (ROWS_TO_EXCLUDE_PER_SIDE>0): img_array = img_array[ROWS_TO_EXCLUDE_PER_SIDE:-ROWS_TO_EXCLUDE_PER_SIDE,:]
        if (COLUMNS_TO_EXCLUDE_PER_SIDE>0): img_array = img_array[:,COLUMNS_TO_EXCLUDE_PER_SIDE:-COLUMNS_TO_EXCLUDE_PER_SIDE]

        #loop over slices and compute the average
        for idx_slice in range(2,d):
            slice = img_array[:,:,idx_slice-1:idx_slice+2]
            if (slice_context=='average'): slice = np.average(slice,axis=2)

            if (output_file_extension=='nii.gz'): 
                nib.save(nib.Nifti1Image(slice,np.eye(4)),f"{output_folder}/{img_name}_slice{idx_slice}.{output_file_extension}")
            elif (output_file_extension=='png')or(output_file_extension=='jpg')or(output_file_extension=='jpeg'): 
                imageio.imwrite(f"{output_folder}/{img_name}_slice{idx_slice}.{output_file_extension}",slice.astype(np.uint8))
            

    typer.echo(f"{len(nifti_files)} 3D nifti scans converted in total.")

@app.command()
def reformat_labels(input_labels_file:str,
                    output_labels_folder:str,
                    overwrite:bool = typer.Option(...,help='Whether to overwrite output files'),
                    input_file_extension: Optional[fileExtension]=typer.Option('nii.gz','--input-extension','-ie', help='Extension of the input files. No other formats implemented yet.'),
                    output_file_extension: Optional[fileExtension]=typer.Option('nii.gz','--output-extension','-oe', help='Extension of the output files. No other formats implemented yet.'),
):
    """Write the CMB position in the correct format for YOLO object detection"""

    if (overwrite): file_access_mode = 'w'
    else: file_acess_mode = 'a'

    labels_original = pd.read_csv(input_labels_file)
    labels_original.sort_values(['NIFTI File Name','z_position'],inplace=True, ignore_index = True)

    slices_files_name = [x.split('/')[-1] for x in iglob(f"{output_labels_folder}/*.{output_file_extension}")]
    nifti_3d_files = np.unique(np.array([x.split('_slice')[:-1] for x in slices_files_name]))
    #start = time.time()
    for file_name in track(nifti_3d_files,description='Writing new labels...'): #for every 3d volume
        file_labels = labels_original[ labels_original['NIFTI File Name']==f"{file_name}.{input_file_extension}"] #object information from that slice  

        current_slices_file_names =  list(filter(lambda k: k.startswith(file_name),slices_files_name))  #all slices from the current volume
    
        slice_number = np.sort(np.array([int(x.split('slice')[-1].split('.')[0]) for x in current_slices_file_names]))
            # iglob(f"{output_labels_folder}/{file_name}_slice*.nii.gz")])) ALTERNATIVE
        
        
        for z in slice_number: #from every slice in the 3d volume
            labels_info = file_labels[file_labels['z_position'].astype(int)==z]  #objects from the current slice

            file_name_to_save =f'{output_labels_folder}/{file_name}_slice{z}' 
            if (labels_info.shape[0]==0):
                #if there are no objects, write an empty file
                open(f'{file_name_to_save}.xml',file_access_mode).close()
            else: 
                #if there are objects, write the labels on the file

                x_center = labels_info['x_position'].to_numpy() - COLUMNS_TO_EXCLUDE_PER_SIDE - 1 
                y_center = labels_info['y_position'].to_numpy() - ROWS_TO_EXCLUDE_PER_SIDE - 1 
                
                x1 = x_center - BOUNDING_BOX_HALF_WIDTH
                x2 = x_center + BOUNDING_BOX_HALF_WIDTH
                y1 = y_center - BOUNDING_BOX_HALF_HEIGHT
                y2 = y_center + BOUNDING_BOX_HALF_HEIGHT

                #write xml annotations
                annot = ET.Element('annotation')

                filename = ET.SubElement(annot,'filename')
                filename.text = file_name

                size = ET.SubElement(annot,'size')
                width = ET.SubElement(size,'width')
                width.text = str(IMAGE_WIDTH - 2*COLUMNS_TO_EXCLUDE_PER_SIDE)
                height = ET.SubElement(size,'height')
                height.text = str(IMAGE_HEIGHT - 2*ROWS_TO_EXCLUDE_PER_SIDE)
                depth = ET.SubElement(size,'depth')
                depth.text = str(1)

                for object_idx in range(labels_info.shape[0]):

                    object = ET.SubElement(annot,'object')
                    name = ET.SubElement(object,'name')
                    name.text = 'microbleed'
                    bndbox = ET.SubElement(object,'bndbox')
                    xmin = ET.SubElement(bndbox,'xmin')
                    xmin.text = str(x1[object_idx])
                    ymin = ET.SubElement(bndbox,'ymin')
                    ymin.text = str(y1[object_idx])
                    xmax = ET.SubElement(bndbox,'xmax')
                    xmax.text = str(x2[object_idx])
                    ymax = ET.SubElement(bndbox,'ymax')
                    ymax.text = str(y2[object_idx])

                xml_file = open(f'{file_name_to_save}.xml','wb')
                xml_file.write(ET.tostring(annot))

                # #transform positions after resizing
                # x = labels_info['x_position'].to_numpy() - COLUMNS_TO_EXCLUDE_PER_SIDE - 1 
                # y = labels_info['y_position'].to_numpy() - ROWS_TO_EXCLUDE_PER_SIDE - 1 

                # slice_width = IMAGE_WIDTH - 2*COLUMNS_TO_EXCLUDE_PER_SIDE
                # slice_height = IMAGE_HEIGHT - 2*ROWS_TO_EXCLUDE_PER_SIDE
                # #scale values 
                # bounding_box_x = x/slice_width
                # bounding_box_y = y/slice_height

                # new_labels = pd.DataFrame({ 'object_class': 'microbleed',
                #                             'x':bounding_box_x,
                #                             'y': bounding_box_y,
                #                             'width': BOUNDING_BOX_WIDTH,
                #                             'height': BOUNDING_BOX_HEIGHT} )
                # new_labels.to_csv(f'{file_name_to_save}.txt',header=None,index=None,sep=' ',mode=file_access_mode)

    #spent_time = (time.time()-start)
    typer.echo(f"Labels from {len(nifti_3d_files)} 3D nifti scans reformatted.")


@app.command()
def create_annotated_slices_YOLO(input_folder:str,
                                output_folder: str,labels_file: str, 
                                overwrite:bool = typer.Option(...,help='Whether to overwrite output files'),
                                slice_context: Optional[SliceContext]=typer.Option('average','--slice-context','-sc', help='Mode to take the context of a slice.'),
                                input_file_extension: Optional[fileExtension]=typer.Option('nii.gz','--input-extension','-ie', help='Extension of the input files. No other formats implemented yet.'),
                                output_file_extension: Optional[fileExtension]=typer.Option('nii.gz','--output-extension','-oe', help='Extension of the output files. No other formats implemented yet.'),
):
    """extract-1channel-slices + reformat-labels.
        Extract averaged slices from 3D nifti files and write label files for YOLO object detection."""
    extract_1channel_slices(input_folder,output_folder,slice_context=slice_context,input_file_extension=input_file_extension,output_file_extension=output_file_extension)
    reformat_labels(labels_file,output_folder,overwrite=overwrite,input_file_extension=input_file_extension,output_file_extension=output_file_extension)

@app.command()
def split_train_test(test_size:str,
                    input_folder:str,
                    output_folder:str,
                    shuffle:Optional[bool] = typer.Option(True,help='Whether to shuffle data before splitting.'),
                    file_extension: Optional[str]=typer.Option('nii.gz',
                    '--extension',
                    '-e',
                    help='Extension of the input files. No other formats implemented yet.')
):
    """Splits train and test data and writes .txt files.
    Only works for linux os."""
    
    #find paths
    paths = np.array([[x,f'{x[:-(len(file_extension)+1)]}.xml'] for x in iglob(f'{input_folder}/*.{file_extension}')])
    #split train and test
    X_train,X_test,y_train,y_test = train_test_split(paths[:,0],paths[:,1],test_size=float(test_size),random_state=42,shuffle=shuffle)

    #make train and test folders 
    os.system(f'mkdir {output_folder}/train')
    os.system(f'mkdir {output_folder}/train/images')
    os.system(f'mkdir {output_folder}/train/annotations')
    os.system(f'mkdir {output_folder}/test')
    os.system(f'mkdir {output_folder}/test/images')
    os.system(f'mkdir {output_folder}/test/annotations')

    #copying splits to the new folder
    for i in track(range(len(X_train)),description='Spliting train and test sets...'):
        copy(X_train[i],f'{output_folder}/train/images')
        copy(y_train[i],f'{output_folder}/train/annotations')

        if (i<len(X_test)): 
            copy(X_test[i],f'{output_folder}/test/images')
            copy(y_test[i],f'{output_folder}/test/annotations')

    
    #saving txt files with the paths
    train_paths = [x for x in iglob(f'{output_folder}/train/images/*.{file_extension}')]
    test_paths = [x for x in iglob(f'{output_folder}/test/images/*.{file_extension}')]
    np.savetxt(f'{output_folder}/train.txt',train_paths,fmt='%s')  
    np.savetxt(f'{output_folder}/test.txt',test_paths,fmt='%s')

    typer.echo(f'Information written in \n{output_folder}/train.txt\n{output_folder}/test.txt')

@app.command()
def preprocess_YOLO(input_folder:str ,
                    output_folder: str,
                    labels_file: str, 
                    test_size:str,
                    overwrite: bool = typer.Option(...,help='Whether to overwrite output files'),
                    slice_context: Optional[SliceContext]=typer.Option('average','--slice-context','-sc', help='Mode to take the context of a slice.'),
                    input_file_extension: Optional[fileExtension]=typer.Option('nii.gz','--input-extension','-ie', help='Extension of the input files. No other formats implemented yet.'),
                    output_file_extension: Optional[fileExtension]=typer.Option('nii.gz','--output-extension','-oe', help='Extension of the output files. No other formats implemented yet.'),
                    shuffle:Optional[bool] = typer.Option(True,help='Whether to shuffle data before splitting.')):
    """extract-1channel-slices + reformat-labels.
        Extract averaged slices from 3D nifti files and write label files for YOLO object detection."""
    
    #create a folder 
    temp_folder = os.popen('mkdir temp_folder; cd temp_folder; pwd').read()[:-1]

    extract_1channel_slices(input_folder,temp_folder,slice_context=slice_context,input_file_extension=input_file_extension,output_file_extension=output_file_extension)
    reformat_labels(labels_file,temp_folder,overwrite=overwrite,input_file_extension=input_file_extension,output_file_extension=output_file_extension)
    split_train_test(test_size=test_size,input_folder=temp_folder,output_folder=output_folder,shuffle=shuffle,file_extension=output_file_extension)

    ouput_delete_folder = os.popen('rm -r temp_folder').read()
    typer.echo(ouput_delete_folder)

if __name__ == "__main__":
    app()