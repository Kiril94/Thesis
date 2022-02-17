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

#from repository
script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parents[1]  #cobra directory
table_dir = join(base_dir, 'tables')
input_file = join(table_dir,'excluded.csv')

#from hdd
disk_dir = "F:" #hdd
dst_data_dir = join(disk_dir,"CoBra","Data")
dcm_dst_data_dir = join(dst_data_dir,"dcm")
nii_excluded_dst_data_dir = join(dst_data_dir,"swi_nii")
nii_included_dst_data_dir =  join(nii_excluded_dst_data_dir,"cmb_study")
dst_log_file = join(dst_data_dir,"log_general_swi.csv")
dst_log_file2 = join(dst_data_dir,"log_downloaded_swi.csv")
dst_log_file3 = join(dst_data_dir,"log_to_download_swi.csv")

#make destination directories if they dont exist
# os.makedirs(dcm_dst_data_dir)
# os.makedirs(nii_included_dst_data_dir)

#from sif
sif_dir = ""


included = False

#%% EXTRACT SCANS INFO + DIRECTORY IN SIF 
df_ids_to_download = pd.read_csv(input_file)
df_scan_info = pd.read_csv(join(table_dir,'swi_all.csv'))

df_info_to_download = df_ids_to_download.merge(df_scan_info,how='inner',left_on='',right_on='PatientID',validate='one_to_one')

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

        #check if it is in the logfile!! 

        #check if it is already downloaded
        origin_dcm_file_path = join(sif_dir,month,row['PatientID'],row['SeriesInstanceUID']) #find out
        dst_dcm_file_path = join(dcm_dst_data_dir,month,row['PatientID'],row['SeriesInstanceUID']) #find out 
        
        #download
        if (not os.path.exists(dst_dcm_file_path)):
            #download 
            shutil.copytree(origin_dcm_file_path,dst_dcm_file_path)
            log_file.write(f"Patient {row['PatientID']} DICOM downloaded")
        
        #check if it is converted
        origin_nii_file_path = join(dst_data_dir,month,row['PatientID'],row['SeriesInstanceUID']) #find out
        if (included): dst_nii_file_path = join(nii_included_dst_data_dir,) #find out     
        else:   dst_nii_file_path = join(nii_excluded_dst_data_dir,) 

        #convert
        if (os.path.exists()):
            #move to swi folder 
            shutil.copytree(origin_nii_file_path,dst_nii_file_path)
            log_file.write(f"Patient {row['PatientID']} nifti moved to swi folder.")
        else: 
            #convert
            dcm2nii_out = dcm2nii.convert_dcm2nii(
                dst_dcm_file_path, dst_nii_file_path, verbose=0, op_sys=0)

            #conversion failed, remove output files
            if dcm2nii_out==1:
                log_file.write(f"Patient {row['PatientID']}: conversion from disk fail")
                log_file3.write(row['PatientID'])

                rm_files = [f for f in os.listdir(dst_nii_file_path) if f.startswith(row['PatientID'])]
                for rm_file in rm_files:
                    remove_file(rm_file)
                
                continue
            # conversion succeed write in log
            else:
                log_file.write(f"Patient {row['PatientID']}: conversion from disk success")

            #fix nifti conversion

        log_file2.write(row['PatientID'])

except:
    log_file.write("!!!!INTERRUPTED!!!!")
    log_file.close()
    log_file2.close()

