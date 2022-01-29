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
#%%
disk_data_dir = join("F:\\", 'CoBra', 'Data')
dcm_base_dir = join(disk_data_dir, 'dcm')
pos_nii_dir = join(disk_data_dir, 'nii', 'positive')
pred_input_dir = join(disk_data_dir, 'volume_longitudinal_nii', 'input')
 
script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parent
data_dir = join(base_dir, 'data')
table_dir = join(data_dir, 'tables')
pat_groups_dir = join(data_dir, 'patient_groups')

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
def get_source_target_dirs(df, base_src_dir, 
            base_tgt_dir):
    return [
        (join(base_src_dir, row.PatientID, row.SeriesInstanceUID+'.nii'),
    join(base_tgt_dir, row.PatientID, row.SeriesInstanceUID+'.nii.gz'))\
    for _, row in df.iterrows()]  


pat_sids_cases_src_tgt_dirs = get_source_target_dirs(
    df_cases, base_src_dir=pos_nii_dir, 
    base_tgt_dir=join(pred_input_dir, 'cases') )
pat_sids_pos_no_cases_src_tgt_dirs = get_source_target_dirs(
    df_pos_no_cases, base_src_dir=pos_nii_dir, 
    base_tgt_dir=join(pred_input_dir, 'positives_cases_excluded') )

src_tgt_ls = pat_sids_cases_src_tgt_dirs + \
    pat_sids_pos_no_cases_src_tgt_dirs


#%%
def log_(str_):
    with open(join(base_dir,"move_files_for_pred_log.txt"), 'a+') as f:
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
    tgt_pat_dir = split(tgt_path)[0]
    if not os.path.isdir(tgt_pat_dir):
        if test:
            log_("Create patient dir at: "+tgt_pat_dir)
        os.mkdir(tgt_pat_dir)
        
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
        nii_out_path = split(src_path)[0]
        # check if dcm path exists
        if os.path.isfile(dcm_path):
            if test:
                start=time.time()
            dcm2nii.convert_dcm2nii(
                dcm_path, nii_out_path, verbose=0, op_sys=0,
                output_filename='%j')
            if test:
                log_("The conversion took "+str(round(time.time()-start,3))+'s')
        else:
            if test:
                log_(dcm_path, 'Does not exist') 
            print('o')
            return 0
        if os.path.isfile(src_path): 
            if test:
                log_('The file was converted to nii and can now be found at '+
                        join(nii_out_path, sid))
            sys.stdout.flush()
            print('+', end='')
            if test:
                log_("The file can be now be moved to "+ tgt_path)
            move_and_gz_files(src_tgt)
        elif len([f for f in os.listdir(split(src_path)[0]) if \
                f.startswith(split(src_path)[1]) and f.endswith('.nii')])>0:
            # Attention! Sometimes the dcm2nii converter produces several files with 
            # different endings like sid_i*.nii
            # We should now call move_and_gz_files on those files
            src_files = [f for f in os.listdir(split(src_path)[0]) if \
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
                        log_('Move all nii files that start with ', split(src_path)[1])
                    # move all files if there is no with ending _i00002.nii
                    tgt_files = [join(split(src_tgt)[0], f) for f in src_files]
                    src_tgt_ls_temp = [(s,t) for s,t in zip(src_files, tgt_files)]
                    for src_tgt_temp in src_tgt_ls_temp:
                        move_and_gz_files(src_tgt_temp)
                    print('*', end='')
            else:
                if test:
                        log_('Move all nii files that start with ', split(src_path)[1])
                tgt_files = [join(split(src_tgt)[0], f) for f in src_files]
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
                current_proc = mp.current_process()    
                current_proc_id = str(int(current_proc._identity[0]))
                write_file = join(
                    pred_input_dir, current_proc_id+'nii_conversion_error_sids.txt')
                with open(write_file,'a+') as f:
                    f.write(dcm_path+'\n')



#%%
def main(source_target_list, procs=8):
    print('file moved: .')
    print('multiple files moved: *')
    print('file converted to nii: +')
    print('No nii, dcm to create nii was not downloaded (yet): o')
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
        for i in range(1000,1010):
            sid_num = i
            move_and_gz_files(src_tgt_ls[sid_num], test=True)
            print(src_tgt_ls[sid_num][0])
            print(src_tgt_ls[sid_num][1])
        print(round(time.time()-start, 3))
    else:
        main(src_tgt_ls, procs=10)