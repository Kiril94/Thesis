"""
created on 15th Feb 2022
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
#%% DIRECTORIES 
def _log(msg,file=None):
    
    if (file is not None): file.write(msg+'\n')
    print(msg)
    
#from repository
script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parents[1]  #cobra directory
table_dir = join(base_dir, 'tables')
input_file = join(table_dir,'ids_swi_excluded.csv')
included = False

#from hdd
disk_dir = "F:" #hdd
dst_data_dir = join(disk_dir,"CoBra","Data")
dcm_dst_data_dir = join(dst_data_dir,"dcm")
nii_excluded_dst_data_dir = join(dst_data_dir,"swi_nii")
nii_included_dst_data_dir =  join(nii_excluded_dst_data_dir,"cmb_study")
dst_log_file = join(dst_data_dir,"log_general_swi.txt")
dst_log_file2 = join(dst_data_dir,"log_downloaded_swi.txt")
dst_log_file3 = join(dst_data_dir,"log_to_download_swi.txt")

# make destination directories if they dont exist
if (not os.path.exists(dcm_dst_data_dir)): os.makedirs(dcm_dst_data_dir)
if (not os.path.exists(nii_included_dst_data_dir)): os.makedirs(nii_included_dst_data_dir)

#from sif
sif_dir = "X:"

#%% EXTRACT SCANS INFO + DIRECTORY IN SIF 
df_ids_to_download = pd.read_csv(input_file)
df_ids_downloaded = pd.read_csv(dst_log_file2)
df_ids_to_download = df_ids_to_download[ ~df_ids_to_download['PatientID'].isin(df_ids_downloaded['PatientID']) ]
df_scan_info = pd.read_csv(join(table_dir,'swi_all.csv'))
df_volume_dir = pd.read_csv(join(table_dir, 'series_directories.csv'))

df_info_to_download = df_ids_to_download.merge(df_scan_info,how='inner',left_on='PatientID',right_on='PatientID',validate='one_to_one')

#open log files
log_file = open(dst_log_file,'a') #general logfile

if (os.path.isfile(dst_log_file2)): #logfile with successes
    log_file2 = open(dst_log_file2,'a') 
else: 
    log_file2 = open(dst_log_file2,'w')
    log_file2.write('PatientID\n')

if (os.path.isfile(dst_log_file3)): #logfile with failures
    log_file3 = open(dst_log_file3,'a') 
else: 
    log_file3 = open(dst_log_file3,'w')
    log_file3.write('PatientID\n')



for idx,row in df_info_to_download.iterrows():

    try: 
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

        #check if it is converted
        origin_nii_file_path = join(dst_data_dir,scan_dir) #find out
        if (included): dst_nii_file_path = join(nii_included_dst_data_dir,scan_dir) #find out     
        else:   dst_nii_file_path = join(nii_excluded_dst_data_dir,scan_dir) 

        #convert
        if (os.path.exists(origin_nii_file_path)):
            #move to swi folder 
            shutil.copytree(origin_nii_file_path,dst_nii_file_path)
            _log(f"Patient {row['PatientID']} nifti moved to swi folder.",log_file)
        else: 
            done = False
            while (not done):
                if check_dicoms(dst_dcm_file_path, origin_dcm_file_path)==0: # check if all the dicoms are on the disk
                    dcm2nii_out = dcm2nii_safe(dst_dcm_file_path, dst_nii_file_path, 
                                            row['PatientID'], True)
                    if dcm2nii_out==0:
                        _log(f"Patient {row['PatientID']}: conversion from disk success",log_file)
                        done = True
                    else:
                        if check_if_philips(dst_dcm_file_path)==0:
                            if fix_dcm_incomplete_vols.fix_incomplete_vols(dst_dcm_file_path)==0:
                                #now we have to adjust the src dir
                                dst_dcm_file_path = join(dst_dcm_file_path, 'corrected_dcm')
                                #start bucle again
                            else:
                                dcm2nii_out = 1
                                done = True
                        else:
                            dcm2nii_out = 1
                            done = True

                    #conversion failed, remove output files
                    if dcm2nii_out==1:
                        _log(f"Patient {row['PatientID']}: conversion from disk fail",log_file)
                        log_file3.write(row['PatientID']+'\n')

                        rm_files = [f for f in os.listdir(dst_nii_file_path) if f.startswith(row['PatientID'])]
                        for rm_file in rm_files:
                            remove_file(rm_file)
                
                else:
                    done = True
                # continue
            # conversion succeed write in log
            else:
                _log(f"Patient {row['PatientID']}: conversion from disk success",log_file)

        log_file2.write(row['PatientID']+'\n')
        _log('||||||||||||||||||||\n',log_file)
        
    except:
        pid = row['PatientID']
        print(f'Patient {pid} raised an error.')
        log_file3.write(pid+'\n')


