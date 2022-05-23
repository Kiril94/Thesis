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
#input_file = join(table_dir,"SWIMatching",'rest_excluded.csv')
#input_file = "C:/Users/neus/Desktop/check_included_scans_v2.txt"
input_file = join(table_dir,"extracted_for_domain_adapt_v3.csv")
included = False

#from hdd
disk_dir = "F:" #hdd
dst_data_dir = join(disk_dir,"CoBra","Data")
dcm_dst_data_dir = join(dst_data_dir,"dcm")
nii_excluded_dst_data_dir = join(dst_data_dir,"swi_nii")
nii_included_dst_data_dir =  join(nii_excluded_dst_data_dir,"cmb_study")
dst_log_file = join(dst_data_dir,"logs","log_general_swi_annotate.txt")
dst_log_file2 = join(dst_data_dir,"logs","log_downloaded_swi_annotate.txt")
dst_log_file3 = join(dst_data_dir,"logs","log_to_download_swi_annotate.txt")
dst_log_file4 = join(dst_data_dir,"logs","log_already_down_swi_annotate.txt")

# make destination directories if they dont exist
if (not os.path.exists(dcm_dst_data_dir)): os.makedirs(dcm_dst_data_dir)
if (not os.path.exists(nii_included_dst_data_dir)): os.makedirs(nii_included_dst_data_dir)

#from sif
sif_dir = "X:"

#%% EXTRACT SCANS INFO + DIRECTORY IN SIF 

df_ids_to_download = pd.read_csv(input_file)

# # do not include already downloaded files
# df_ids_downloaded = pd.read_csv(dst_log_file2)
# df_ids_to_download = df_ids_to_download[ ~df_ids_to_download['PatientID'].isin(df_ids_downloaded['PatientID']) ]
# # do not include already failed files
# df_ids_failed = pd.read_csv(dst_log_file3)
# df_ids_to_download = df_ids_to_download[ ~df_ids_to_download['PatientID'].isin(df_ids_failed['PatientID']) ]

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

if (os.path.isfile(dst_log_file4)): #logfile with failures
    log_file4 = open(dst_log_file4,'a') 
else: 
    log_file4 = open(dst_log_file4,'w')
    log_file4.write('PatientID\n')

print(f"{df_info_to_download.shape[0]} DICOM files to download.")


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

        #check if all files were downloaded
        else:
            files_in_origin = next(os.walk(origin_dcm_file_path))[2]
            files_in_dst = next(os.walk(dst_dcm_file_path))[2]
            
            if (len(files_in_origin)>len(files_in_dst)):
                
                missing_files = filter(lambda x: x not in files_in_dst, files_in_origin)
                
                print("Downloading missing files")
                for file in missing_files:
                    shutil.copyfile(join(origin_dcm_file_path,file),join(dst_dcm_file_path,file))
                    
        #check if it is converted
        origin_nii_file_path = join(dst_data_dir,scan_dir) #find out
        if (included): dst_nii_file_path = join(nii_included_dst_data_dir,scan_dir) #find out     
        else:   dst_nii_file_path = join(nii_excluded_dst_data_dir,scan_dir) 

        #convert
        done = False
        if (os.path.exists(origin_nii_file_path)):
            origin_no_phase_files = list(filter(lambda x: not x.endswith('_ph.nii.gz'),next(os.walk(origin_nii_file_path))[2]))
            if len(origin_no_phase_files)>0:
                #move to swi folder 
                shutil.copytree(origin_nii_file_path,dst_nii_file_path)
                _log(f"Patient {row['PatientID']} nifti moved to swi folder.",log_file)
                done = True
                    
        elif (os.path.exists(dst_nii_file_path)):
            dst_no_phase_files = list(filter(lambda x: not x.endswith('_ph.nii.gz'),next(os.walk(dst_nii_file_path))[2]))     
            if len(dst_no_phase_files)>0:
                #file already downloaded
                _log(f"Patient {row['PatientID']} already downloaded.",log_file)
                log_file4.write(row['PatientID']+'\n')
                done = True
        if not done: 
            while (not done):
                if check_dicoms(dst_dcm_file_path, origin_dcm_file_path)==0: # check if all the dicoms are on the disk
                    dcm2nii_out = dcm2nii_safe(dst_dcm_file_path, dst_nii_file_path, 
                                            row['PatientID'], True)
                    if dcm2nii_out==0:
                        _log(f"Patient {row['PatientID']}: conversion from disk success",log_file)
                        print("*******************************")
                        print(next(os.walk(dst_nii_file_path))[2])
                        print("*******************************")
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
                    #download again
                    shutil.copytree(origin_dcm_file_path,dst_dcm_file_path)
                    _log(f"Patient {row['PatientID']} DICOM downloaded",log_file)
                    done = False
                # continue
            # conversion succeed write in log
            else:
                _log(f"Patient {row['PatientID']}: conversion from disk success",log_file)

        log_file2.write(row['PatientID']+'\n')
        _log('||||||||||||||||||||\n',log_file)
    
    except KeyboardInterrupt:
        print('KeyBoard interrrupt')
        break    
    except:
        pid = row['PatientID']
        print(f'Patient {pid} raised an error.')
        log_file3.write(pid+'\n')


log_file.close()
log_file2.close()
log_file3.close()