"""
created on march 21st
author: Neus Rodeja Ferrer
"""

#%%
import os
from pathlib import Path
from os.path import join
import pandas as pd 
import shutil
import nibabel as nib
import numpy as np

#from repository
script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parents[1]  #cobra directory
table_dir = join(base_dir, 'tables')

#from hdd
disk_dir = "F:" #hdd
dst_data_dir = join(disk_dir,"CoBra","Data")
origin_nii_data_dir = join(dst_data_dir,"swi_nii")
dst_nii_data_dir = join(dst_data_dir,"swi_nii","test_pipeline","volumes")
log_file_path = join(dst_data_dir,"swi_nii","test_pipeline",'names_dict.csv')

sif_dir = "Y:"
#%%
# Taking first 5 with high prob of having CMB from excluded

df_excl_probs = pd.read_csv(join(table_dir,"ids_swi_excluded_pcmb_v3.csv"))
series_directories_df = pd.read_csv(join(table_dir,'series_directories.csv'))
info_swi_all = pd.read_csv(join(table_dir,'swi_all.csv'))

info_excluded = info_swi_all.merge(df_excl_probs,how='inner',on='PatientID',validate='one_to_one')
info_excluded = info_excluded.merge(series_directories_df,how='inner',on='SeriesInstanceUID',validate='one_to_one')

high_5 = info_excluded.iloc[[1,3,4,5,14,16,18]]
low_5 = info_excluded.tail(11)

#%% reanem
log_file = open(log_file_path,'a')
log_file.write('SeriesInstanceUID,old_path,new_name\n')

idx_name=0
for idx,row in high_5.iterrows():
    dir = row['Directory']
    path = join(origin_nii_data_dir,dir)
    suid = row['SeriesInstanceUID']
    
    try:
        files_in_orig_path =  [f for f in os.listdir(path) if os.path.isfile(join(path, f))]
    except FileNotFoundError as e:
        print('IDX',idx)
        print(e)
        continue
    
    name = 'H' + str(idx_name).zfill(3) + '.nii.gz'
    dst_path = join(dst_nii_data_dir,name)
    
    if (len(files_in_orig_path)==1):
        shutil.copy(join(path,files_in_orig_path[0]),dst_path)
        log_file.write(f'{suid},{dir},{name}\n')
        idx_name += 1
            
    elif (len(files_in_orig_path)==2):
        filename1 = files_in_orig_path[0]
        filename2 = files_in_orig_path[1].split('_')
                    
        if (filename1[:-7] == filename2[0])&(filename2[1]=='ph.nii.gz'):
            #copy both files
            shutil.copy(join(path,filename1),dst_path)
            shutil.copy(join(path,files_in_orig_path[1]),join(dst_nii_data_dir,'phases',name[:-7]+'_ph.nii.gz'))
            log_file.write(f'{suid},{dir},{name}\n')
            idx_name += 1
        
        else:    
            filename1 = files_in_orig_path[0].split('_') 
            filename2 = files_in_orig_path[1]
            if (filename1[0] == filename2[:-7])&(filename1[1]=='ph.nii.gz'):
                #copy both files
                shutil.copy(join(path,filename2),dst_path)
                shutil.copy(join(path,files_in_orig_path[0]),join(dst_nii_data_dir,'phases',name[:-7]+'_ph.nii.gz'))        
                log_file.write(f'{suid},{dir},{name}\n')
                idx_name += 1
            
    elif (len(files_in_orig_path)>2):
        print(f'More than 2 files in SeriesInstanceUID {suid}')
    else:
        print(f'No files found in SeriesInstanceUID {suid}')

#%% 
# rename low
print("MOVING TO LOW...........")
idx_name=0
for idx,row in low_5.iterrows():
    dir = row['Directory']
    path = join(origin_nii_data_dir,dir)
    suid = row['SeriesInstanceUID']
    
    try:
        files_in_orig_path =  [f for f in os.listdir(path) if os.path.isfile(join(path, f))]
    except FileNotFoundError as e:
        print(e)
        continue
    
    name = 'L' + str(idx_name).zfill(3) + '.nii.gz'
    dst_path = join(dst_nii_data_dir,name)
    
    if (len(files_in_orig_path)==1):
        shutil.copy(join(path,files_in_orig_path[0]),dst_path)
        log_file.write(f'{suid},{dir},{name}\n')
        idx_name += 1
            
    elif (len(files_in_orig_path)==2):
        filename1 = files_in_orig_path[0]
        filename2 = files_in_orig_path[1].split('_')
                    
        if (filename1[:-7] == filename2[0])&(filename2[1]=='ph.nii.gz'):
            #copy both files
            shutil.copy(join(path,filename1),dst_path)
            shutil.copy(join(path,files_in_orig_path[1]),join(dst_nii_data_dir,'phases',name[:-7]+'_ph.nii.gz'))
            log_file.write(f'{suid},{dir},{name}\n')
            idx_name += 1
        
        else:    
            filename1 = files_in_orig_path[0].split('_') 
            filename2 = files_in_orig_path[1]
            if (filename1[0] == filename2[:-7])&(filename1[1]=='ph.nii.gz'):
                #copy both files
                shutil.copy(join(path,filename2),dst_path)
                shutil.copy(join(path,files_in_orig_path[0]),join(dst_nii_data_dir,'phases',name[:-7]+'_ph.nii.gz'))        
                log_file.write(f'{suid},{dir},{name}\n')
                idx_name += 1
            
    elif (len(files_in_orig_path)>2):
        print(f'More than 2 files in SeriesInstanceUID {suid}')
    else:
        print(f'No files found in SeriesInstanceUID {suid}')
        
log_file.close()

#%%
# slice volumes

origin_volumes_path = dst_nii_data_dir
dst_slices_path = join(dst_data_dir,"swi_nii","test_pipeline","slices")

files = next(os.walk(origin_volumes_path))[2]

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

for file in files:
    
    volume_data, _ = load_nifti_img(join(origin_volumes_path,file))
    n_slices = volume_data.shape[2]
    
    for i_slice in range(n_slices):
        slice_data = volume_data[:,:,i_slice]
        new_name = file[:-7] + f'_slice{i_slice}' + '.nii.gz'
        slice_path = join(dst_slices_path,new_name)
        
        nib.save(nib.Nifti1Image(slice_data,np.eye(4)),slice_path)
        