"""
created on 15th Feb 2022
author: neus rodeja ferrer
"""

import os
from pathlib import Path
from os.path import join
import pandas as pd 
import shutil
from dcm2nii import dcm2nii
from basic import remove_file
#%% DIRECTORIES 
def _log(msg,file=None):
    
    if (file is not None): file.write(msg)
    print(msg)
    
#from repository
script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parents[1]  #cobra directory
table_dir = join(base_dir, 'tables')
input_file = join(table_dir,'excluded.csv')
included = False

#from hdd
disk_dir = "F:" #hdd
dst_data_dir = join(disk_dir,"CoBra","Data")
dcm_dst_data_dir = join(dst_data_dir,"dcm")
nii_excluded_dst_data_dir = join(dst_data_dir,"swi_nii")
nii_included_dst_data_dir =  join(nii_excluded_dst_data_dir,"cmb_study")
dst_log_file = join(dst_data_dir,"log_general_swi.csv")
dst_log_file2 = join(dst_data_dir,"log_downloaded_swi.csv")
dst_log_file3 = join(dst_data_dir,"log_to_download_swi.csv")

# make destination directories if they dont exist
os.makedirs(dcm_dst_data_dir)
os.makedirs(nii_included_dst_data_dir)

#from sif
sif_dir = "X:"

#%% EXTRACT SCANS INFO + DIRECTORY IN SIF 
#df_ids_to_download = pd.read_csv(input_file)
df_ids_to_download = pd.DataFrame({'PatientID': ['7f01474ed0460f8f9c1ce78b348b9728']
                                  })
df_scan_info = pd.read_csv(join(table_dir,'swi_all.csv'))
df_volume_dir = pd.read_csv(join(table_dir, 'series_directories.csv'))

df_info_to_download = df_ids_to_download.merge(df_scan_info,how='inner',left_on='PatientID',right_on='PatientID',validate='one_to_one')

#open log files
log_file = open(dst_log_file,'a') #general logfile

if (os.path.isfile(dst_log_file2)): #logfile with successes
    log_file2 = open(dst_log_file2,'a') 
else: 
    log_file2 = open(dst_log_file2,'w')
    log_file2.write('PatientID')

if (os.path.isfile(dst_log_file3)): #logfile with failures
    log_file3 = open(dst_log_file3,'a') 
else: 
    log_file3 = open(dst_log_file3,'w')
    log_file3.write('PatientID')


try: 
    for idx,row in df_info_to_download.iterrows():

        scan_dir = df_volume_dir[ df_volume_dir['SeriesInstanceUID']==row['SeriesInstanceUID'] ]['Directory'].values[0]
        #check if it is in the logfile!! 

        #check if it is already downloaded
        origin_dcm_file_path = join(sif_dir,scan_dir) #find out
        dst_dcm_file_path = join(dcm_dst_data_dir,scan_dir) #find out 
        
        #download
        if (not os.path.exists(dst_dcm_file_path)):
            #download 
            shutil.copytree(origin_dcm_file_path,dst_dcm_file_path)
            _log(f"Patient {row['PatientID']} DICOM downloaded",log_file)

        #check if all files were downloaded
        else:
            files_in_origin = next(os.walk(origin_dcm_file_path))[2]
            files_in_dst = next(os.walk(dst_dcm_file_path))[2]
            
            if (len(files_in_origin)>len(files_in_dst)):
                
                missing_files = filter(lambda x: x not in files_in_dst, files_in_origin)
                for file in missing_files:
                    shutil.copyfile(join(origin_dcm_file_path,file),join(dst_dcm_file_path,file))
                    
        #check if it is converted
        origin_nii_file_path = join(dst_data_dir,scan_dir) #find out
        if (included): dst_nii_file_path = join(nii_included_dst_data_dir,) #find out     
        else:   dst_nii_file_path = join(nii_excluded_dst_data_dir,) 

        #convert
        if (os.path.exists(origin_nii_file_path)):
            #move to swi folder 
            shutil.copytree(origin_nii_file_path,dst_nii_file_path)
            _log(f"Patient {row['PatientID']} nifti moved to swi folder.",log_file)
        else: 
            #convert
            dcm2nii_out = dcm2nii.convert_dcm2nii(
                dst_dcm_file_path, dst_nii_file_path, verbose=0, op_sys=0)

            #conversion failed, remove output files
            if dcm2nii_out==1:
                _log(f"Patient {row['PatientID']}: conversion from disk fail",log_file)
                log_file3.write(row['PatientID'])

                rm_files = [f for f in os.listdir(dst_nii_file_path) if f.startswith(row['PatientID'])]
                for rm_file in rm_files:
                    remove_file(rm_file)
                
                continue
            # conversion succeed write in log
            else:
                _log(f"Patient {row['PatientID']}: conversion from disk success",log_file)

            #fix nifti conversion

        log_file2.write(row['PatientID'])
        _log('||||||||||||||||||||\n',log_file)
except:
    log_file.write("!!!!INTERRUPTED!!!!")
    log_file.close()
    log_file2.close()

