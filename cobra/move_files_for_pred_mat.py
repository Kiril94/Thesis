#%%
import shutil
import os
from os.path import join, split
from pathlib import Path
import multiprocessing as mp
import time
import json
import pickle
from utilities.basic import list_subdir, make_dir, get_proc_id
import matlab.engine
from functools import partial
# paths
script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parent
tables_dir = join(base_dir, 'data', 'tables')
disk_data_dir = join("F:\\", 'CoBra', 'Data')
tgt_dir = join(disk_data_dir, 'volume_longitudinal_nii', 'input', 'nii_files')
tmp_dir = join(disk_data_dir, 'volume_longitudinal_nii', 'temp')
# matlab engine
eng = matlab.engine.start_matlab()
eng.addpath('C:\\Users\\kiril\\Thesis\\CoBra\\cobra\\dcm2nii\\dcm2nii_mat\\functions', nargout=0)
eng.addpath('C:\\Users\\kiril\\Thesis\\CoBra\\cobra\\dcm2nii\\dcm2nii_mat\\dcm2nii')
eng.addpath('C:\\Users\\kiril\\Thesis\\CoBra\\cobra\\dcm2nii\\dcm2nii_mat\\spm12')
# load necessary files
with open(join(tables_dir, 'newIDs_dic.pkl'), 'rb') as f:
    id_dic = pickle.load(f)
with open("C:\\Users\\kiril\\Thesis\\CoBra\\cobra\\data\\t1_longitudinal\\pairs_3dt1_long_sids.pkl", 'rb') as f:
    sids_ls = pickle.load(f)
with open("C:\\Users\\kiril\\Thesis\\CoBra\\cobra\\data\\tables\\disk_series_directories.json", 'rb') as f:
    dir_dic = json.load(f)

# define functions
def get_missing_files(sids_to_conv, nii_dir, newid_dic):
    """
    sids_to_conv: List of SeriesInstanceUIDs that need to be converted to nii
    nii_dir: str, directory where converted files are placed
    newid_dic: dictionary used to map sids to 6 digit new ids
    returns: list of missing files sids
    """
    inv_map = {v: k for k, v in newid_dic.items()}
    conv_files_ids = [file[:-7] for file in os.listdir(nii_dir)]
    conv_files_sids = [inv_map[id] for id in conv_files_ids]
    missing_files = set(sids_to_conv).difference(set(conv_files_sids))
    return list(missing_files)
    
def dcm2nii_mat(src_dir, tgt_path, tmp_dir, test=False):
    """Converts dcm to nii using dcm2nii (matlab) or spm12 (matlab) if first fails
    src_dir: Directory with dcm series
    tgt_path: Full path of the nii file that will be produced (should end with .nii.gz)"""
    tmp_dir_sp = join(tmp_dir, str(get_proc_id(test)))
    make_dir(tmp_dir_sp)
    try:
        eng.dcm2nii_main(src_dir, tmp_dir_sp)
    except:
        shutil.remove()
        print("dcm2nii failed, try spm")
        try:
            eng.spm12_main(src_dir, tmp_dir_sp)
        except:
            for f in list_subdir(tmp_dir_sp):
                shutil.remove(f)
            print('x')
    out_files = list_subdir(tmp_dir_sp, ending='.nii.gz')
    assert len(out_files)<=1, f'More than 1 nii file was created for {src_dir}'
    if len(out_files)==1:
        shutil.move(out_files[0], tgt_path)
    else:
        pass
    return 0
def dcm2nii_mat_main(sids_ls, id_dic, tmp_dir, tgt_dir, test=False):
    """sids_ls: List of sids that need to be converted"""
    missing_files = get_missing_files(sids_ls, tgt_dir, id_dic)
    if test:
        missing_files = missing_files[10:12]
    sids = [split(f)[1] for f in missing_files]
    tgt_paths = [join(tgt_dir, id_dic[sid]+'.nii.gz') for sid in sids]
    src_dirs = [dir_dic[sid] for sid in sids]

    mp_input = [(src_dir, tgt_path) for src_dir, tgt_path in zip(src_dirs, tgt_paths)]
    dcm2nii_mat_p = partial(dcm2nii_mat, tmp_dir=tmp_dir)
    with mp.Pool() as pool:
                pool.starmap(dcm2nii_mat_p, mp_input)

if __name__ == '__main__':
    dcm2nii_mat_main(sids_ls, id_dic, tmp_dir, tgt_dir, test=True)