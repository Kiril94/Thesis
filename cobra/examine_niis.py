import os
from os.path import join, split
from pathlib import Path
import matlab.engine
import pickle
import json
from utilities.basic import list_subdir, remove_file, remove_files
import shutil
script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parent

disk_dir = "F:"
disk_data_dir = join(disk_dir, 'CoBra', 'Data')
dcm_data_dir = join(disk_data_dir,'dcm')
data_dir = join(base_dir, 'data')
tables_dir = join(data_dir, 'tables')
log_corr_dir = join(disk_data_dir, 'volume_longitudinal_nii', 'input')
tgt_dir = join(disk_data_dir, 'volume_longitudinal_nii', 'input', 'nii_files')
tmp_dir = join(disk_data_dir, 'volume_longitudinal_nii', 'temp')
#cutoff_dir = join(tmp_dir, 'spm_conv_error', 'cut_off')
#cutoff_newids = [split(f)[1][:-7] for f in os.listdir(cutoff_dir)]
with open("C:\\Users\\kiril\\Thesis\\CoBra\\cobra\\data\\t1_longitudinal\\sids_long_stroke.pkl", 'rb') as f:
    sids = pickle.load(f) 
with open(join(tables_dir, 'newIDs_dic.pkl'), 'rb') as f:
    id_dic = pickle.load(f)
with open("C:\\Users\\kiril\\Thesis\\CoBra\\cobra\\data\\t1_longitudinal\\sids_long_new.pkl", 'rb') as f:
    sids_ls = pickle.load(f)
with open("C:\\Users\\kiril\\Thesis\\CoBra\\cobra\\data\\tables\\disk_series_directories.json", 'rb') as f:
    dir_dic = json.load(f)
inv_id_map = {v: k for k, v in id_dic.items()}
#dcm_dirs = {newid:os.path.normpath(dir_dic[inv_id_map[newid]]) for newid in newids}
dcm_dirs = {sid:os.path.normpath(dir_dic[sid]) for sid in sids}
for item in dcm_dirs.items():
    print(item)
