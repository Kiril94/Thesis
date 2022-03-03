"""
created on 3rd March 2022
author: neus rodeja ferrer
"""

import os
from pathlib import Path
from os.path import join
import pandas as pd 
import shutil
from cobra.dcm2nii import dcm2nii
from cobra.download_data.basic import remove_file
from cobra.download_data import check_dicoms,check_if_philips,fix_dcm_incomplete_vols,dcm2nii_safe
# DIRECTORIES 
def _log(msg,file=None):
    
    if (file is not None): file.write(msg+'\n')
    print(msg)
    
#from repository
script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parents[1]  #cobra directory
table_dir = join(base_dir, 'tables')
input_file = join(table_dir,'ids_swi_included.csv')
included = False

#from hdd
disk_dir = "F:" #hdd
dst_data_dir = join(disk_dir,"CoBra","Data")
nii_excluded_dst_data_dir = join(dst_data_dir,"swi_nii")
nii_included_dst_data_dir =  join(nii_excluded_dst_data_dir,"cmb_study")
orig_folder = nii_excluded_dst_data_dir
dst_folder = join(nii_included_dst_data_dir,"nii")
dst_log_file = join(nii_included_dst_data_dir,"log_renamed_swi.txt")
dst_log_file2 = join(nii_included_dst_data_dir,"log_failed_renaming_swi.txt")
dst_log_file3 = join(nii_included_dst_data_dir,"log_not_previously_down_swi.txt")
# make destination directories if they dont exist
if (not os.path.exists(dst_folder)): os.makedirs(dst_folder)

#from sif
sif_dir = "X:"

#initialize log files
if (os.path.isfile(dst_log_file2)): 
    log_file = open(dst_log_file,'a') 
else: 
    log_file = open(dst_log_file,'w')
    log_file.write('SeriesInstanceUID,old_path,new_name\n')
    
if (os.path.isfile(dst_log_file2)): 
    log_file2 = open(dst_log_file2,'a') 
else: 
    log_file2 = open(dst_log_file2,'w')
    log_file2.write('SeriesInstanceUID\n')

if (os.path.isfile(dst_log_file3)): 
    log_file3 = open(dst_log_file3,'a') 
else: 
    log_file3 = open(dst_log_file3,'w')
    log_file3.write('SeriesInstanceUID\n')
        
#%% EXTRACT SCANS INFO + DIRECTORY IN SIF 
df_ids_to_rename = pd.read_csv(input_file)
df_scan_info = pd.read_csv(join(table_dir,'swi_all.csv'))
df_volume_dir = pd.read_csv(join(table_dir, 'series_directories.csv'))

df_info_to_rename = df_ids_to_rename.merge(df_scan_info,how='inner',left_on='PatientID',right_on='PatientID',validate='one_to_one')
df_info_to_rename = df_info_to_rename.merge(df_volume_dir,how='inner',left_on='SeriesInstanceUID',right_on='SeriesInstanceUID',validate='one_to_one')
df_info_to_rename.sort_values(by='Directory',inplace=True)

name_prev = ''
n_vols_from_folder = 1
for idx,row in df_info_to_rename.iterrows():
    
    try:
        # find path
        dir = row['Directory']
        orig_path  = join(orig_folder,dir)
        suid = row['SeriesInstanceUID']
        
        #check that it is downloaded
        if (not os.path.exists(orig_path)):         
            suid = row['SeriesInstanceUID']
            print(f'Scan {suid} not downloaded.')
            log_file3.write(suid+'\n')
            continue
        
        
        # define name 
        month_folder = dir.split('\\')[0]
        #start with the month for month folder, and 20 for positive folder
        if (len(month_folder.split('_'))>1):
            name = month_folder.split('_')[-1]
        else: 
            name = '20'
            
        # if we change folder, restart the n_vols_from_folder
        if (name_prev[:2]!=name):
            n_vols_from_folder = 1            
        name = name + str(n_vols_from_folder).zfill(4) + '.nii.gz'
        
        # move file 
        dst_path = join(dst_folder,name)
        files_in_orig_path =  [f for f in os.listdir(orig_path) if os.path.isfile(join(orig_path, f))]
        
        if (len(files_in_orig_path)==1):
            shutil.copy(join(orig_path,files_in_orig_path[0]),dst_path)
            log_file.write(f'{suid},{dir},{name}\n')
            n_vols_from_folder += 1
            name_prev = name
            
        elif (len(files_in_orig_path)==2):
            filename1 = files_in_orig_path[0]
            filename2 = files_in_orig_path[1].split('_')
                        
            if (filename1[:-7] == filename2[0])&(filename2[1]=='ph.nii.gz'):
                #copy both files
                shutil.copy(join(orig_path,filename1),dst_path)
                shutil.copy(join(orig_path,files_in_orig_path[1]),join(dst_folder,'phases',name[:-7]+'_ph.nii.gz'))
                log_file.write(f'{suid},{dir},{name}\n')
                n_vols_from_folder += 1
                name_prev = name
            
            else:    
                filename1 = files_in_orig_path[0].split('_') 
                filename2 = files_in_orig_path[1]
                if (filename1[0] == filename2[:-7])&(filename1[1]=='ph.nii.gz'):
                    #copy both files
                    shutil.copy(join(orig_path,filename2),dst_path)
                    shutil.copy(join(orig_path,files_in_orig_path[0]),join(dst_folder,'phases',name[:-7]+'_ph.nii.gz'))        
                    log_file.write(f'{suid},{dir},{name}\n')
                    n_vols_from_folder += 1
                    name_prev = name
                
        elif (len(files_in_orig_path)>2):
            print(f'More than 2 files in SeriesInstanceUID {suid}')
            log_file2.write(suid+'\n')
        else:
            print(f'No files found in SeriesInstanceUID {suid}')
            log_file2.write(suid+'\n')      

    except Exception as e:
        print(e)
        print(f'Scan {suid} raised an error.')
        log_file2.write(suid+'\n')
        log_file2.write(str(e) + '\n\n')
        
        
log_file.close()
log_file2.close()
log_file3.close()