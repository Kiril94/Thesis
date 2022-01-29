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
dfc = pd.read_csv(join(table_dir, "neg_pos_clean.csv"), 
    usecols=['SeriesInstanceUID', 'PatientID', 'MRAcquisitionType',
    'Sequence', 'NumberOfSlices'])
cases_ls = np.loadtxt(join(pat_groups_dir, 
                't1_pre_post.txt'), dtype=str)
#%%
df3dt1 = dfc[(dfc.MRAcquisitionType=='3D') & \
    (dfc.Sequence=='t1') & (dfc.NumberOfSlices>=64)]
df_cases = df3dt1[df3dt1.PatientID.isin(cases_ls)]
df_pos_no_cases = df3dt1[~df3dt1.PatientID.isin(cases_ls)]

# %%
def get_root(path, n=2):
    return join(*os.path.normpath(path).split(os.sep)[:n])

def get_source_target_dirs(df, base_src_dir, 
            base_tgt_dir):
    return [
        (join(base_src_dir, get_root(volume_dir_dic[sid]), split(volume_dir_dic[sid])[1] +'.nii'),
    join(base_tgt_dir, split(get_root(volume_dir_dic[sid]))[1], split(volume_dir_dic[sid])[1] +'.nii.gz'))\
    for sid in df.SeriesInstanceUID]  
def write_problematic_files(file):
    current_proc = mp.current_process()    
    current_proc_id = str(int(current_proc._identity[0]))
    write_file = join(
            pred_input_dir, current_proc_id+'nii_conversion_error_sids.txt')
    with open(write_file,'a+') as f:
        f.write(file+'\n')
def dcm2nii_safe(dcm_path, nii_out_path, sid, test):
    "Only keep dicoms if dcm2nii converter returns 0"
    dcm2nii_out = dcm2nii.convert_dcm2nii(
        dcm_path, nii_out_path, verbose=0, op_sys=0,
                output_filename='%j', create_info_json='y')
    if dcm2nii_out==1: #if dcm2nii produces error, remove all the output files
        rm_files = [f for f in os.listdir(nii_out_path) if f.startswith(sid)]
        for rm_file in rm_files:
            remove_file(rm_file)
        if not test:
            write_problematic_files(dcm_path)

pat_sids_cases_src_tgt_dirs = get_source_target_dirs(
    df_cases, base_src_dir=disk_nii_dir, 
    base_tgt_dir=join(pred_input_dir, 'cases') )
pat_sids_pos_no_cases_src_tgt_dirs = get_source_target_dirs(
    df_pos_no_cases, base_src_dir=disk_nii_dir, 
    base_tgt_dir=join(pred_input_dir, 'positives_cases_excluded') )

src_tgt_ls = pat_sids_cases_src_tgt_dirs + \
    pat_sids_pos_no_cases_src_tgt_dirs


#%%
def log_(str_):
    with open(join(base_dir, "move_files_for_pred_log.txt"), 'a+') as f:
        f.write(str_+'\n')
def move_and_gz_files(src_tgt, test=False):
    sys.stdout.flush()
    src_path = src_tgt[0]
    tgt_path = src_tgt[1]
    if os.path.isfile(tgt_path):
        if test:
            log_("The file already exists at "+tgt_path)
            log_('Stop')
        return 0
    # create patient dir
    tgt_pat_dir = get_dir(tgt_path)
    make_dir(tgt_pat_dir)
    # if nii file exists move and gz compress it    
    if os.path.isfile(src_path): 
        if test:
            log_("Nii file exists at " + src_path)
            log_("Move and gz compress that file")
        with open(src_path, 'rb') as f_in:
            with gzip.open(tgt_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        sys.stdout.flush()
        print(".", end='')

    else: # if nii does not exist, create it
        if test:
            log_("Nii file does NOT exist at "+ src_path)
            log_("Try to convert dcm to nii")
        month_dir, pid, sid = os.path.normpath(src_path).split(os.sep)[-3:]
        sid = sid[:-4] #remove .nii extension 
        dcm_path = join(disk_data_dir, 'dcm', month_dir, pid, sid)
        nii_out_path = get_dir(src_path)
        # check if dcm path exists
        if os.path.isdir(dcm_path):
            make_dir(get_dir(nii_out_path))
            if test:
                log_(dcm_path + " exists, start nii conversion")
                start=time.time()
            dcm2nii_safe(dcm_path, nii_out_path, sid, test)
            if test:
                log_("The conversion took "+str(round(time.time()-start,3))+'s')
        else: #if dcm path doesn't exist make conversion directly from sif
            dcm_path = join(sif_dir, volume_dir_dic[sid])
            make_dir(get_dir(nii_out_path))
            if os.path.isdir(dcm_path):
                if test:
                    log_(get_dir(dcm_path) + ' Does not exist on disk')
                    log_('Convert directly from sif') 
                    start=time.time()            
                dcm2nii_safe(dcm_path, nii_out_path, sid, test)
                if test:
                    log_("The conversion took "+str(round(time.time()-start,3))+'s')
                print('o', end='')
            else:
                if test:
                    log_(dcm_path+ ' Does not exist on sif')
                print('-', end='')
                return 0
        make_dir(get_dir(src_path))
        if os.path.isfile(src_path): 
            if test:
                log_('The file was converted to nii and can now be found at '+
                        join(nii_out_path, sid))
            sys.stdout.flush()
            print('+', end='')
            if test:
                log_("The file can be now be moved to "+ tgt_path)
            move_and_gz_files(src_tgt)
        elif len([f for f in os.listdir(get_dir(src_path)) if \
                f.startswith(split(src_path)[1]) and f.endswith('.nii')])>0:
            # Attention! Sometimes the dcm2nii converter produces several files with 
            # different endings like sid_i*.nii
            # We should now call move_and_gz_files on those files
            src_files = [f for f in os.listdir(get_dir(src_path)) if \
                f.startswith(split(src_path)[1]) and f.endswith('.nii')]
            # if there are two nii the endings are usually i00001.nii and i00002.nii
            # the first is localizer and second is actual 3d image
            if len(src_files)==2:
                if test:
                    log_('There are two nii files for this sid')
                src_file_temp = [f for f in src_files if f.endswith('_i00002.nii')]
                if len(src_file_temp)==1:
                    if test:
                        log_('Copy the one with ending _i00002.nii')
                    move_and_gz_files((src_file_temp[0], tgt_path))
                else:
                    if test:
                        log_('Move all nii files that start with '+ split(src_path)[1])
                    # move all files if there is no with ending _i00002.nii
                    tgt_files = [join(get_dir(src_tgt), f) for f in src_files]
                    src_tgt_ls_temp = [(s,t) for s,t in zip(src_files, tgt_files)]
                    for src_tgt_temp in src_tgt_ls_temp:
                        move_and_gz_files(src_tgt_temp)
                    print('*', end='')
            else:
                if test:
                        log_('Move all nii files that start with '+ split(src_path)[1])
                tgt_files = [join(get_dir(src_tgt), f) for f in src_files]
                src_tgt_ls_temp = [(s,t) for s,t in zip(src_files, tgt_files)]
                for src_tgt_temp in src_tgt_ls_temp:
                    move_and_gz_files(src_tgt_temp)
                print('*', end='')
        else: #if some issue with nii conversion skip this file
            sys.stdout.flush()
            print('x')
            if test:
                log_('dcm2nii failed')
            else:
                write_problematic_files(dcm_path)



#%%
def main(source_target_list, procs=8):
    print('file moved: .')
    print('multiple files moved: *')
    print('file converted to nii: +')
    print('No nii, dcm to create nii was not downloaded (yet), convert directly from SIF: o')
    print('dcm does not exist at all: -')
    print('fail: x')
    print("Move ", len(src_tgt_ls), "files.")
    print(f"Using {procs} processes")
    with mp.Pool(procs) as pool:
                pool.map(move_and_gz_files, 
                        source_target_list)
    print(dt.now())

if __name__ == '__main__':
    test=True
    if test:
        print('Test')
        start = time.time()
        for i in range(1010,1012):
            sid_num = i
            move_and_gz_files(src_tgt_ls[sid_num], test=True)
            print(src_tgt_ls[sid_num][0])
            print(src_tgt_ls[sid_num][1])
        print(round(time.time()-start, 3))
    else:
        main(src_tgt_ls, procs=10)