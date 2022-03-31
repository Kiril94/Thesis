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

#from repository
script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parents[1]  #cobra directory
table_dir = join(base_dir, 'tables')

#from hdd
disk_dir = "F:" #hdd
dst_data_dir = join(disk_dir,"CoBra","Data")
origin_nii_excl_data_dir = join(dst_data_dir,"swi_nii")
origin_nii_incl_data_dir = join(dst_data_dir,"swi_nii","cmb_study")
dst_nii_data_dir = join(dst_data_dir,"swi_nii","cmb_study","new_nii")
log_file_path = join(dst_data_dir,"swi_nii","cmb_study","names_new_nii.csv")


#%%
# Taking paths

df_new = pd.read_csv(join(table_dir,"ids_swi_included_new_v3.csv"))
series_directories_df = pd.read_csv(join(table_dir,'series_directories.csv'))
info_swi_all = pd.read_csv(join(table_dir,'swi_all.csv'))

df_new_info = info_swi_all.merge(df_new,how='inner',on='PatientID',validate='one_to_one')
df_new_info = df_new_info.merge(series_directories_df,how='inner',on='SeriesInstanceUID',validate='one_to_one')

#%%
#copy and rename files

log_file = open(log_file_path,'w')
log_file.write('SeriesInstanceUID,old_path,new_name\n')

idx_name = 0
for idx,row in df_new_info.iterrows():
    dir = row['Directory']
    suid = row['SeriesInstanceUID']
    
    path_inc = join(origin_nii_incl_data_dir,dir)
    path_exc = join(origin_nii_excl_data_dir,dir)
    
    try:
        files_in_inc_folder = next(os.walk(path_inc))[2]
    except:
        files_in_inc_folder = []
    
    try:
        files_in_exc_folder = next(os.walk(path_exc))[2] 
    except:
        files_in_exc_folder = []
    
    if len(files_in_exc_folder)>0:
        
        path = path_exc
        files_in_folder = files_in_exc_folder

        name = 'NEW' + str(idx_name).zfill(4) + '.nii.gz'
        dst_path = join(dst_nii_data_dir,name)
        
        if (len(files_in_folder)==1):
            shutil.copy(join(path,files_in_folder[0]),dst_path)
            log_file.write(f'{suid},{dir},{name}\n')
            idx_name += 1
                
        elif (len(files_in_folder)==2):
            filename1 = files_in_folder[0]
            filename2 = files_in_folder[1].split('_')
                        
            if (filename1[:-7] == filename2[0])&(filename2[1]=='ph.nii.gz'):
                #copy both files
                shutil.copy(join(path,filename1),dst_path)
                shutil.copy(join(path,files_in_folder[1]),join(dst_nii_data_dir,'phases',name[:-7]+'_ph.nii.gz'))
                log_file.write(f'{suid},{dir},{name}\n')
                idx_name += 1
            
            else:    
                filename1 = files_in_folder[0].split('_') 
                filename2 = files_in_folder[1]
                if (filename1[0] == filename2[:-7])&(filename1[1]=='ph.nii.gz'):
                    #copy both files
                    shutil.copy(join(path,filename2),dst_path)
                    shutil.copy(join(path,files_in_folder[0]),join(dst_nii_data_dir,'phases',name[:-7]+'_ph.nii.gz'))        
                    log_file.write(f'{suid},{dir},{name}\n')
                    idx_name += 1
                
        elif (len(files_in_folder)>2):
            print(f'More than 2 files in SeriesInstanceUID {suid}')
            
    elif len(files_in_inc_folder)>0:
        
        path = path_inc
        files_in_folder = files_in_inc_folder
        
        name = 'NEW' + str(idx_name).zfill(4) + '.nii.gz'
        dst_path = join(dst_nii_data_dir,name)
        
        if (len(files_in_folder)==1):
            shutil.copy(join(path,files_in_folder[0]),dst_path)
            log_file.write(f'{suid},{dir},{name}\n')
            idx_name += 1
                
        elif (len(files_in_folder)==2):
            filename1 = files_in_folder[0]
            filename2 = files_in_folder[1].split('_')
                        
            if (filename1[:-7] == filename2[0])&(filename2[1]=='ph.nii.gz'):
                #copy both files
                shutil.copy(join(path,filename1),dst_path)
                shutil.copy(join(path,files_in_folder[1]),join(dst_nii_data_dir,'phases',name[:-7]+'_ph.nii.gz'))
                log_file.write(f'{suid},{dir},{name}\n')
                idx_name += 1
            
            else:    
                filename1 = files_in_folder[0].split('_') 
                filename2 = files_in_folder[1]
                if (filename1[0] == filename2[:-7])&(filename1[1]=='ph.nii.gz'):
                    #copy both files
                    shutil.copy(join(path,filename2),dst_path)
                    shutil.copy(join(path,files_in_folder[0]),join(dst_nii_data_dir,'phases',name[:-7]+'_ph.nii.gz'))        
                    log_file.write(f'{suid},{dir},{name}\n')
                    idx_name += 1
                
        elif (len(files_in_folder)>2):
            print(f'More than 2 files in SeriesInstanceUID {suid}')

    else:
        print(f'No files found in SeriesInstanceUID {suid}')
        
        
        
#%%
#