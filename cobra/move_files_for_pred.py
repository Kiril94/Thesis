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

#%%
disk_data_dir = join("F:\\", 'CoBra', 'Data')
dcm_base_dir = join(disk_data_dir, 'dcm')
pos_nii_dir = join(disk_data_dir, 'nii', 'positive')
pred_input_dir = join(disk_data_dir, 'volume_longitudinal', 'input')

script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parent
data_dir = join(base_dir, 'data')
table_dir = join(data_dir, 'tables')
pat_groups_dir = join(data_dir, 'patient_groups')

dfc = pd.read_csv(join(table_dir, "neg_pos_clean.csv"))
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
    base_tgt_dir=join(pred_input_dir, 'positive_cases_excluded') )

src_tgt_ls = pat_sids_cases_src_tgt_dirs + \
    pat_sids_pos_no_cases_src_tgt_dirs

print("Move ", len(src_tgt_ls), "files.")
#%%
def move_and_gz_files(src_tgt):
    sys.stdout.flush()
    src_path = src_tgt[0]
    tgt_path = src_tgt[1]
    if os.path.isfile(tgt_path):
        return 0
    # create patient dir
    tgt_pat_dir = split(tgt_path)[0]
    if not os.path.isdir(tgt_pat_dir):
        os.mkdir(tgt_pat_dir)
    if os.path.isfile(src_path): 
        with open(src_path, 'rb') as f_in:
            with gzip.open(src_tgt[1], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(".", end='')
    else: # if nii does not exist, create it
        month_dir, pid, sid = os.path.normpath(src_path).split(os.sep)[-3:]
        sid = sid[:-4] #remove .nii extension 
        dcm_path = join(disk_data_dir, 'dcm', month_dir, pid)
        nii_out_path = split(src_path)[0]
        print('dicom path', dcm_path)
        print('nii path', nii_out_path)
        dcm2nii.convert_dcm2nii(
            dcm_path, nii_out_path, verbose=0, op_sys=0,
            output_filename='%j')
        if os.path.isfile(src_path): 
            print('+',end='')
            move_and_gz_files(src_tgt)
        else: #if some issue with nii conversion skip this file
            current_proc = mp.current_process()    
            current_proc_id = str(int(current_proc._identity[0]))
            with open(join(pred_input_dir, current_proc_id+'nii_conversion_error_sids.txt'),'a+') as f:
                f.write(sid)




#%%
def main(source_target_list, procs=8):
    with mp.Pool(procs) as pool:
                pool.map(move_and_gz_files, 
                        source_target_list)
    print(dt.now())

if __name__ == '__main__':
    test=False
    if test:
        print('Test')
        #print(src_tgt_ls[0])
        main(src_tgt_ls[:80], procs=8)
    else:
        main(src_tgt_ls, procs=10)