#%%
import shutil
import sys
import os
from os.path import join, split
from pathlib import Path
import pandas as pd
import numpy as np
import gzip
import multiprocessing as mp
from dcm2nii import dcm2nii
from datetime import datetime as dt
import time
import json
import pickle
from utilities import basic
from utilities.basic import get_dir, make_dir, remove_file
#%%
disk_data_dir = join("F:\\", 'CoBra', 'Data')
dcm_base_dir = join(disk_data_dir, 'dcm')
disk_nii_dir = join(disk_data_dir, 'nii')
pred_input_dir = join(disk_data_dir, 'volume_longitudinal_nii', 'input')
sif_dir = 'Y:\\'
script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parent
data_dir = join(base_dir, 'data')
table_dir = join(data_dir, 'tables')
pat_groups_dir = join(data_dir, 'patient_groups')

df_volume_dir = pd.read_csv(join(table_dir, 'series_directories.csv'))
volume_dir_dic = pd.Series(
    df_volume_dir.Directory.values, index=df_volume_dir.SeriesInstanceUID)\
        .to_dict()
with open(join(table_dir, "disk_series_directories.json"), "r") as json_file:
    disk_volume_dir_dic = json.load(json_file)
dfc = pd.read_csv(join(table_dir, "neg_pos_clean.csv"), 
    usecols=['SeriesInstanceUID', 'PatientID', 'MRAcquisitionType',
    'Sequence', 'NumberOfSlices'])

sids_3d_t1_path = join(data_dir, 't1_longitudinal', 'pairs_3dt1_longitudinal_study.pkl')
with open(sids_3d_t1_path, 'rb') as f:
    sids_3dt1_long = pickle.load(f)
sids_cases = np.loadtxt(join(pat_groups_dir, 
                't1_pre_post_suid.txt'), dtype=str).tolist()
df_cases_controls = dfc[dfc.SeriesInstanceUID.isin(sids_3dt1_long)]
df_cases = df_cases_controls[df_cases_controls.SeriesInstanceUID.isin(sids_cases)]
df_controls = df_cases_controls[~(df_cases_controls.SeriesInstanceUID.isin(sids_cases))]
# %%
def get_root_dir(path, n=2):
    return join(*os.path.normpath(path).split(os.sep)[:n])

def get_source_target_dirs(df, base_src_dir, 
            base_tgt_dir):
    return [
        (join(base_src_dir, get_root_dir(volume_dir_dic[sid]), split(volume_dir_dic[sid])[1] +'.nii'),
    join(base_tgt_dir, split(get_root_dir(volume_dir_dic[sid]))[1], split(volume_dir_dic[sid])[1] +'.nii.gz'))\
    for sid in df.SeriesInstanceUID]  
def get_proc_id(test=False):
    if test:
        return 0
    else:
        current_proc = mp.current_process()    
        current_proc_id = str(int(current_proc._identity[0]))
        return current_proc_id
def write_problematic_files(file, test):
    if test:
        return 0
    current_proc_id = get_proc_id(test)
    write_file = join(
            pred_input_dir, 'logs', 
            current_proc_id+'nii_conversion_error_sids.txt')
    with open(write_file,'a+') as f:
        f.write(file+'\n')

def move_compress(src, tgt):
    with open(src, 'rb') as f_in:
                with gzip.open(tgt, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

def dcm2nii_safe(disk_dcm_path, sif_dcm_path, nii_out_path, sid, test, trial, timeout=2000):
    "Only keep dicoms if dcm2nii converter returns 0"
    if trial<=1 and os.path.isdir(disk_dcm_path):
        if test:
                log_(disk_dcm_path + " exists, start nii conversion")
                start=time.time()
        print(get_proc_id(test), " Convert from disk")
        dcm2nii_out = dcm2nii.convert_dcm2nii(
            disk_dcm_path, nii_out_path, verbose=0, op_sys=0,
                    output_filename='%j', create_info_json='y', timeout=timeout)
        if test:
                log_("The conversion took "+str(round(time.time()-start,3))+'s')
                if dcm2nii_out==1:
                    log_("conversion from disk fail")
                else:
                    log_("conversion from disk success")
        
        if dcm2nii_out==0:
            print(get_proc_id(test), " worked")
            return 0
        elif dcm2nii_out==1: #if dcm2nii produces error, remove all the output files
            if not test:
                    write_problematic_files(disk_dcm_path, test)
            if len(os.listdir(disk_dcm_path))==len(os.listdir(sif_dcm_path)):
                print(get_proc_id(test), "Error from conversion on disk, but same data on sif")
                return 0
            else:
                print(get_proc_id(test), "There is more data on sif, use this!")
                rm_files = [f for f in os.listdir(nii_out_path) if f.startswith(sid)]
                for rm_file in rm_files:
                    remove_file(rm_file)
                    return 1 
    else:
        print(get_proc_id(test), " convert from sif")
        if not os.path.isdir(sif_dcm_path):
            write_problematic_files(sif_dcm_path, test)
            print("DCM missing on sif")
            return 1
        if test:
                log_('dicoms for ' + sid + ' do not exist on disk')
                log_('Convert directly from sif') 
                start=time.time()            
        dcm2nii_out = dcm2nii.convert_dcm2nii(
            sif_dcm_path, nii_out_path, verbose=0, op_sys=0,
                    output_filename='%j', create_info_json='y', timeout=timeout)
        if test:
                log_("The conversion took "+str(round(time.time()-start,3))+'s')
                if dcm2nii_out==1:
                    log_("Conversion from sif error, write file path to problematic files")
                else:
                    log_("Conversion from sif success") 
        if dcm2nii_out==0:
            print(get_proc_id(test), " worked")
            return 0
        elif dcm2nii_out==1: #if dcm2nii from sif produces error, still keep the output file
            print('/')
            if not test:
                write_problematic_files(disk_dcm_path, test)
            print(get_proc_id(test), " SIF conversion gave error for  ", sif_dcm_path)
            return 2
    
        

def summarize_problematic_files():
    dir_ = join(pred_input_dir, 'logs')
    error_log_files = [f for f in basic.list_subdir(dir_) \
        if f.endswith("error_sids.txt")]
    string = '\n'
    for error_log_file in error_log_files:
        with open(error_log_file, 'r') as f:
            text = f.read() + '\n'
        string+=text
    with open(join(dir_, 'nii_conversion_error_sids_all.txt'), 'a+') as f:
        f.write(string)

pat_sids_cases_src_tgt = get_source_target_dirs(
    df_cases, base_src_dir=disk_nii_dir, 
    base_tgt_dir=join(pred_input_dir, 'cases') )
pat_sids_potential_controls_src_tgt = get_source_target_dirs(
    df_controls, base_src_dir=disk_nii_dir, 
    base_tgt_dir=join(pred_input_dir, 'potential_controls') )
print("Convert only potential controls now")
src_tgt_ls =  pat_sids_potential_controls_src_tgt #+ pat_sids_cases_src_tgt

#%%
def check_niis(existing_src_files, src_dir, tgt_path, test, trial):
    if trial==0:
        if test:
            log_("Following nii file(s) were found " + str(existing_src_files))
            log_("Move and gz compress these files")
        if len(existing_src_files)==1: # if only 1 file it is probably the 3d vol.
            sys.stdout.flush()
            print(".", end='')
            move_compress(join(src_dir, existing_src_files[0]), tgt_path)
            return 0
        elif len(existing_src_files)==2: # if 2 files, the one with _i00002.nii is probably the 3d vol.
            files3d = [f for f in existing_src_files if f.endswith("_i00002.nii")]
            if len(files3d)==1:
                sys.stdout.flush()
                print(".", end='')
                move_compress(join(src_dir, files3d[0]), tgt_path)
            else: # otherwise remove all the files and call the function again
                for ex_src_file in existing_src_files:
                    remove_file(join(src_dir, ex_src_file))
                return 1
        else: # if more than 2 files, also remove them
            for ex_src_file in existing_src_files:
                    remove_file(join(src_dir, ex_src_file))
            return 1
    else:
        if len(existing_src_files)==1:
            move_compress(join(src_dir, existing_src_files[0]), tgt_path)
        else:
            for i, ex_src_file in enumerate(existing_src_files):
                tgt_path_tmp = tgt_path[:-7] + '_' + str(i) + '.nii.gz'
                move_compress(join(src_dir, ex_src_file), tgt_path_tmp)
        sys.stdout.flush()
        print(".", end='')
        return 0

def log_(str_):
    with open(join(base_dir, "move_files_for_pred_log.txt"), 'a+') as f:
        f.write(str_+'\n')

def check_tgt_files(tgt_path, sid):
    tgt_dir = get_dir(tgt_path)
    if len([f for f in os.listdir(tgt_dir) if f.startswith(sid)])>0:
        return True
    else:
        return False


def move_and_gz_files(src_tgt, test=False, trial=0):
    if test:
        log_("trial number "+ str(trial))
    if trial>2:
        return 1
    sys.stdout.flush()
    src_path = src_tgt[0]
    month_dir, pid, sid = os.path.normpath(src_path).split(os.sep)[-3:] #we will need it later
    sid = sid[:-4] #remove .nii extension
    tgt_path = src_tgt[1]
    tgt_pat_dir = get_dir(tgt_path)
    make_dir(tgt_pat_dir)
    if check_tgt_files(tgt_path, sid):
        print('|', end='')
        if test:
            log_("The file(s) already exists at " + tgt_path)
            log_('Stop')
        return 0
    print(get_proc_id(test), " Trial: ", trial, " sid: ", sid)
    # create patient dir
    make_dir(get_dir(src_path))
    src_dir = get_dir(src_path)
    
    sif_dcm_path = join(sif_dir, volume_dir_dic[sid])
    disk_dcm_path = disk_volume_dir_dic[sid]
    nii_out_path = get_dir(src_path)
    make_dir(nii_out_path)
    # Handle the case if at least one nii file already exists on the disk
    existing_src_files = [f for f in os.listdir(src_dir) \
        if f.startswith(sid) and f.endswith('.nii')]
    if len(existing_src_files)>0 and trial==0: 
        print(get_proc_id(test), " Check existing niis")
        check_niis_out = check_niis(existing_src_files, src_dir, tgt_path, test, trial)
        trial+=1
        if check_niis_out==1:
            move_and_gz_files(src_tgt, trial=trial)
    else: # if nii does not exist, try to create it
        print(get_proc_id(test), " No niis")
        if test:
            log_("Nii file does NOT exist at "+ src_path)
        # check if dcm dir exists
        dcm2nii_out = dcm2nii_safe(disk_dcm_path, sif_dcm_path, nii_out_path, 
                                sid, test, trial=trial)
        trial+=1     
        if dcm2nii_out==1:
            move_and_gz_files(src_tgt, trial=trial)
        else:
            existing_src_files = [f for f in os.listdir(src_dir) \
                if f.startswith(sid) and f.endswith('.nii')]
            if len(existing_src_files)>0:
                check_niis_out = check_niis(existing_src_files, src_dir, tgt_path, test, trial)            
            else:
                print(get_proc_id(test), " Conversion did not work for", disk_dcm_path, nii_out_path)
                sys.stdout.flush()
                print('x')
                if test:
                    log_('dcm2nii failed')
                else:
                    write_problematic_files(disk_dcm_path, test)
                return 1


                


#%%
def main(source_target_list, procs=8):
    print('file moved: .')
    print('file exists: |')
    print('file converted to nii: +')
    print('fail: x')
    print("Move ", len(src_tgt_ls), "files.")
    print(f"Using {procs} processes")
    with mp.Pool(procs) as pool:
                pool.map(move_and_gz_files, 
                        source_target_list)
    summarize_problematic_files()
    print("Finished at: ", dt.now())
    

if __name__ == '__main__':
    test=False
    if test:
        print('Test')
        summarize_problematic_files()
        start = time.time()
        for i in range(1000,1004):
            sid_num = i
            move_and_gz_files(src_tgt_ls[sid_num], test=True)
        print("Finished at: ", dt.now())
        print("Total time: ",round(time.time()-start, 3))
    else:
        main(src_tgt_ls, procs=10)