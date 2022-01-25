#%%
import shutil
import os
from os.path import join, split
from pathlib import Path
import pandas as pd
import numpy as np
import gzip
import multiprocessing as mp
from dcm2nii import dcm2nii
import importlib
importlib.reload(dcm2nii)
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
    join(base_tgt_dir, row.PatientID+row.SeriesInstanceUID+'.nii.gz'))\
    for _, row in df.iterrows()]  


pat_sids_cases_src_tgt_dirs = get_source_target_dirs(
    df_cases, base_src_dir=pos_nii_dir, 
    base_tgt_dir=join(pred_input_dir, 'cases') )
pat_sids_pos_no_cases_src_tgt_dirs = get_source_target_dirs(
    df_pos_no_cases, base_src_dir=pos_nii_dir, 
    base_tgt_dir=join(pred_input_dir, 'positive_cases_excluded') )

src_tgt_ls = pat_sids_cases_src_tgt_dirs + \
    pat_sids_pos_no_cases_src_tgt_dirs


#%%
def move_and_gz_files_or_create_gz_file(src_tgt):
    src_path = src_tgt[0]
    if os.path.isdir(src_path): 
        return 0
        with open(src_path, 'rb') as f_in:
            with gzip.open(src_tgt[1], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    
    else: # if nii does not exist, create it
        print(src_tgt_ls)
        month_dir, pid, sid = os.path.normpath(src_path).split(os.sep)[-3:]
        sid = sid[:-4] #remove .nii extension 
        dcm_path = join(disk_data_dir, 'dcm', month_dir, pid, sid)
        nii_out_path = src_path
        dcm2nii.convert_dcm2nii(
            dcm_path, nii_out_path, verbose=0, op_sys=0,
            output_filename='%j', gz_compress='n')
        move_and_gz_files_or_create_gz_file(src_tgt)

move_and_gz_files_or_create_gz_file(src_tgt_ls[0])


#%%
def main(source_target_list, procs=8):
    with mp.Pool(procs) as pool:
                pool.map(move_and_gz_files, 
                        source_target_list)

if __name__ == '__main__':
    main(src_tgt_ls[1:5], procs=2)