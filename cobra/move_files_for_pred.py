#%%
import shutil
import os
from os.path import join
from pathlib import Path
import pandas as pd
import numpy as np
import gzip
import multiprocessing as mp
from dcm2nii import dcm2nii

#%%
disk_data_dir = join("F:\\", 'CoBra', 'Data')
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
dcm2nii.convert_dcm2nii_help()
#%%
def move_and_gz_files(src_tgt):
        with open(src_tgt[0], 'rb') as f_in:
            with gzip.open(src_tgt[1], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

def main(source_target_list, procs=8):
    with mp.Pool(procs) as pool:
                pool.map(move_and_gz_files, 
                        source_target_list)

if __name__ == '__main__':
    main(src_tgt_ls[1:5], procs=2)