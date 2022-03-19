#%%
import shutil
import os
from os.path import join, split
from pathlib import Path
import json
import pickle
from utilities.basic import list_subdir, move_compress, remove_files
import matlab.engine
import numpy as np

# paths
script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parent
tables_dir = join(base_dir, 'data', 'tables')
disk_data_dir = join("F:\\", 'CoBra', 'Data')
tgt_dir = join(disk_data_dir, 'volume_cross_nii', 'input', 'nii_files')
tmp_dir = join(disk_data_dir, 'volume_cross_nii', 'temp')
#excl_files_dir = join(tmp_dir, 'spm_conv_error', 'cut_off')
excl_files_dir = [join(disk_data_dir, 'volume_longitudinal_nii', 'input', 'nii_files'), 
    join(disk_data_dir, 'volume_cross_nii', 'input', 'nii_files')]
data_dir = join(base_dir, 'data')
data_cross_dir = join(data_dir, 't1_cross')
# matlab engine
eng = matlab.engine.start_matlab()
eng.addpath('C:\\Users\\kiril\\Thesis\\CoBra\\cobra\\dcm2nii\\dcm2nii_mat\\functions', nargout=0)
eng.addpath('C:\\Users\\kiril\\Thesis\\CoBra\\cobra\\dcm2nii\\dcm2nii_mat\\dcm2nii')
eng.addpath('C:\\Users\\kiril\\Thesis\\CoBra\\cobra\\dcm2nii\\dcm2nii_mat\\spm12')
# load necessary files
with open(join(tables_dir, 'newIDs_dic.pkl'), 'rb') as f:
    id_dic = pickle.load(f)
with open(join(data_cross_dir, "3dt1_sids.pkl"), 'rb') as f:
    sids_ls = pickle.load(f)
with open(join(tables_dir, "disk_series_directories.json"), 'rb') as f:
    dir_dic = json.load(f)
downloaded_sids = np.loadtxt(join(disk_data_dir,'dcm', 'volume_log.txt'), dtype=str).tolist()
sids_ls = list(set(sids_ls).intersection(set(downloaded_sids)))



# define functions
def get_missing_files(sids_to_conv, nii_dir, newid_dic, excl_nii_dir=None):
    """
    sids_to_conv: List of SeriesInstanceUIDs that need to be converted to nii
    nii_dir: str, directory where converted files are placed
    newid_dic: dictionary used to map sids to 6 digit new ids
    returns: list of missing files sids
    """
    inv_map = {v: k for k, v in newid_dic.items()}
    conv_files_ids = [file[:-7] for file in os.listdir(nii_dir)]
    conv_files_sids = [inv_map[id] for id in conv_files_ids]
    if not isinstance(excl_nii_dir, type(None)):
        print('exclude files in', excl_nii_dir)
        if isinstance(excl_nii_dir, list):
            excl_files_sids = []
            for dir_ in excl_nii_dir:
                excl_files_ids = [file[:-7] for file in os.listdir(dir_)]
                excl_files_sids_temp = [inv_map[id] for id in excl_files_ids]
                excl_files_sids = excl_files_sids + excl_files_sids_temp
        else:
            excl_files_ids = [file[:-7] for file in os.listdir(excl_nii_dir)]
            excl_files_sids = [inv_map[id] for id in excl_files_ids]
    missing_files = (set(sids_to_conv).difference(set(conv_files_sids))).difference(set(excl_files_sids))
    return list(missing_files)


def dcm2nii_mat(src_dir, tgt_path, tmp_dir, test=False):
    """Converts dcm to nii using dcm2nii (matlab) or spm12 (matlab) if first fails
    src_dir: Directory with dcm series
    tgt_path: Full path of the nii file that will be produced (should end with .nii.gz)"""
    
    try:
        eng.spm12_main(src_dir, tmp_dir)
    except:
        # sometimes .nii files are produced that look reasonable
        # rename them and keep them in these folder
        nii_files = list_subdir(tmp_dir, '.nii')
        if len(nii_files)==1:
            move_compress(nii_files[0], join(tmp_dir, 'spm_conv_error', split(nii_files[0])[1]+'.gz'), True)
        remove_files(tmp_dir, ending='.nii.gz')
        remove_files(tmp_dir, ending='.nii')
        print("spm failed, try dcm2nii")
        try:
            eng.dcm2nii_main(src_dir, tmp_dir)
        except:
            nii_files = list_subdir(tmp_dir, '.nii')
            if len(nii_files)==1:
                shutil.move(nii_files[0], join(tmp_dir, 'dcm2nii_conv_error', split(tgt_path)[1][:-3]))
            remove_files(tmp_dir, ending='.nii.gz')
            remove_files(tmp_dir, ending='.nii')
            print('x')
    out_files = list_subdir(tmp_dir, ending='.nii.gz')
    if len(out_files)==0:
        pass
    elif len(out_files)==1:
        shutil.move(out_files[0], tgt_path)
    else:
        for out_file in out_files:
            shutil.move(out_file, join(tmp_dir, 'dcm2nii_conv_error', split(out_file)[1]))
    return 0
def dcm2nii_mat_main(sids_ls, id_dic, tmp_dir, tgt_dir, excl_files_dir=None, test=False):
    """sids_ls: List of sids that need to be converted"""
    missing_files = get_missing_files(sids_ls, tgt_dir, id_dic, excl_files_dir)
    print(len(missing_files), ' files will be converted')
    if test:
        missing_files = missing_files[:3]
    sids = [split(f)[1] for f in missing_files]
    tgt_paths = [join(tgt_dir, id_dic[sid]+'.nii.gz') for sid in sids]
    src_dirs = [dir_dic[sid] for sid in sids]
    mp_input = [(src_dir, tgt_path) for src_dir, tgt_path in zip(src_dirs, tgt_paths)]
    for src_dir, tgt_path in mp_input:
        dcm2nii_mat(src_dir, tgt_path, tmp_dir)
    
if __name__ == '__main__':
    dcm2nii_mat_main(sids_ls, id_dic, tmp_dir, tgt_dir, excl_files_dir, test=False)