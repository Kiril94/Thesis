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
logfile = join(tmp_dir, 'conv_log.txt')
excl_files_dir = [join(tmp_dir, 'spm_conv_error', 'cut_off'), 
                join(tmp_dir, 'spm_conv_error', 'inc_imageorientation'),
                ]
# read ids of corrupted files
with open(join(log_corr_dir, 'Corrupted.txt'), 'r') as f:
    lines = []
    for line in f:
        lines.append(line)
corr_new_ids = []
for i, line in enumerate(lines):
        if i<(len(lines)-1):
            corr_new_ids.append(split(line[:-2])[1][:-7])
        else:
            corr_new_ids.append(split(line)[1][:-8])


with open(join(log_corr_dir, 'Corrupted2.txt'), 'r') as f:
    lines = []
    for line in f:
        lines.append(line)

for i, line in enumerate(lines):
        if i<(len(lines)-1):
            corr_new_ids.append(line[-15:-9])
        else:
            corr_new_ids.append(line[-13:-7])


corr_new_ids = corr_new_ids + ['109983', '047205', '083072', '109644', '287546', '258402',
                '258387', '258391', '258394','258397','258402', '258403']

print('Check ids: ', all([len(id)==6 for id in corr_new_ids]))

# exclude_ids = []
# with open(join(log_corr_dir, 'exclude_files.txt'), 'r') as f:
    # for line in f:
        # exclude_ids.append(line[:6])

# print('Exclude ', len(exclude_ids), 'files in', (join(log_corr_dir, 'exclude_files.txt')))
# print('Excluded files', exclude_ids)
# corr_new_ids = list(set(corr_new_ids).difference(set(exclude_ids)))

def remove_corr_files():
    print('Remove corrupted niis')
    for corr_new_id in corr_new_ids:
        file = corr_new_id+'.nii.gz'
        #print(file)
        remove_file(join(tgt_dir, file))
#assert False


eng = matlab.engine.start_matlab()
eng.addpath('C:\\Users\\kiril\\Thesis\\CoBra\\cobra\\dcm2nii\\dcm2nii_mat\\functions', nargout=0)
eng.addpath('C:\\Users\\kiril\\Thesis\\CoBra\\cobra\\dcm2nii\\dcm2nii_mat\\dcm2nii')
eng.addpath('C:\\Users\\kiril\\Thesis\\CoBra\\cobra\\dcm2nii\\dcm2nii_mat\\spm12')
# load necessary files
with open(join(tables_dir, 'newIDs_dic.pkl'), 'rb') as f:
    id_dic = pickle.load(f)
with open("C:\\Users\\kiril\\Thesis\\CoBra\\cobra\\data\\t1_longitudinal\\3dt1_sids.pkl", 'rb') as f:
    sids_ls = pickle.load(f)
with open(join(tables_dir, "disk_series_directories.json"), 'rb') as f:
    dir_dic = json.load(f)

# define functions
def log_(s):
    with open(logfile, 'w') as f:
        f.write(s)

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
        print('Exclude files that are in', excl_nii_dir)
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
    else:
        missing_files = (set(sids_to_conv).difference(set(conv_files_sids)))
    return list(missing_files)

def dcm2nii_mat(src_dir, tgt_path, tmp_dir, test=False):
    """Converts dcm to nii using dcm2nii (matlab) or spm12 (matlab) if first fails
    src_dir: Directory with dcm series
    tgt_path: Full path of the nii file that will be produced (should end with .nii.gz)"""
    print(src_dir)
    print('->', tgt_path)
    tmp_dir_sp = tmp_dir#join(tmp_dir, str(get_proc_id(test)))
    #make_dir(tmp_dir_sp)

    try:
        eng.dcm2nii_main(src_dir, tmp_dir_sp)
        
    except:
        # sometimes .nii files are produced that look reasonable
        # rename them and keep them in this folder
        nii_files = list_subdir(tmp_dir_sp, '.nii')
        if len(nii_files)==1:
            shutil.move(nii_files[0], join(tmp_dir_sp, 'dcm2nii_conv_error', split(tgt_path)[1]))
        remove_files(tmp_dir_sp, ending='.nii.gz')
        print("dcm2nii failed, try som")
        log_('dcm2nii failed on '+ split(tgt_path)[1]+ '\n')
        try:
            eng.spm12_main(src_dir, tmp_dir_sp)
        except:
            log_('spm failed on '+ split(tgt_path)[1]+'\n')
            remove_files(tmp_dir_sp, ending='.nii.gz')
            print('x')
    out_files = list_subdir(tmp_dir_sp, ending='.nii.gz')
    assert len(out_files)<=1, f'More than 1 nii file was created for {src_dir}'
    if len(out_files)==1:
        shutil.move(out_files[0], tgt_path)
    else:
        pass
    print('remove all remaining nii files')
    remove_files(tmp_dir_sp, ending='.nii')
    return 0
def dcm2nii_mat_main(sids_ls, id_dic, tmp_dir, tgt_dir, excl_files_dir=None, test=False):
    """sids_ls: List of sids that need to be converted"""
    print("Get not converted files")
    missing_files = get_missing_files(sids_ls, tgt_dir, id_dic, excl_files_dir)
    print(len(missing_files), ' files will be converted')
    if test:
        missing_files = missing_files[1:2]
    sids = [split(f)[1] for f in missing_files]
    tgt_paths = [join(tgt_dir, id_dic[sid]+'.nii.gz') for sid in sids]
    src_dirs = [dir_dic[sid] for sid in sids]
    mp_input = [(src_dir, tgt_path) for src_dir, tgt_path in zip(src_dirs, tgt_paths)]
    for src_dir, tgt_path in mp_input:
        dcm2nii_mat(src_dir, tgt_path, tmp_dir)

if __name__ == '__main__':
    dcm2nii_mat_main(sids_ls, id_dic, tmp_dir, tgt_dir, excl_files_dir=None, test=False)