#%%
from utilities import utils
from dcm2nii import dcm2nii
import glob
from os.path import join, realpath, split
from pathlib import Path
import shutil
import os

#%%
script_dir = realpath(__file__)
base_dir = Path(script_dir).parent
table_dir = join(base_dir, 'data', 'tables')
target_dirs = join(base_dir, 'data', 't1_longitudinal', 'images')
disk_dir = "G:/"
src_dir = join(disk_dir, 'Cobra', 'Data', 'nii', 'positive')
#%%
df_pc = utils.load_scan_csv(join(table_dir, 't1', 't1_pc_3d.csv'))

df_b = df_pc[df_pc.days_since_test<=-30].sort_values(
    by=['PatientID','SeriesInstanceUID'])#before
df_a = df_pc[df_pc.days_since_test>=-3].sort_values(
    by=['PatientID','SeriesInstanceUID'])#after
#%% copy after
for patient_id in df_a.PatientID.unique():
    df_a_pat = df_a[df_a.PatientID==patient_id] 
    for vol_num, (index, row) in enumerate(df_a_pat.iterrows()):
        src_paths = glob.iglob(join(src_dir, patient_id,
                                    f'{row.SeriesInstanceUID}*.nii'))
        for i, src_path in enumerate(src_paths):
            tgt_name = patient_id + '_' + row.SeriesInstanceUID \
                + '_' + str(i) + '.nii' 
            tgt_path = join(target_dirs, 'after', tgt_name)
            if not os.path.exists(tgt_path):
                shutil.copy(src_path, tgt_path)
#%% 
for patient_id in df_b.PatientID.unique():
    df_b_pat = df_b[df_b.PatientID==patient_id] 
    for vol_num, (index, row) in enumerate(df_b_pat.iterrows()):
        src_paths = glob.iglob(join(src_dir, patient_id,
                                    f'{row.SeriesInstanceUID}*.nii'))
        for i, src_path in enumerate(src_paths):
            tgt_name = patient_id + '_' + row.SeriesInstanceUID \
                + '_' + str(i) + '.nii' 
            tgt_path = join(target_dirs, 'before', tgt_name)
            if not os.path.exists(tgt_path):
                shutil.copy(src_path, tgt_path)