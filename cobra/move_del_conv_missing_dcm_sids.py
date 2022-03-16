"""Delete existing niis, delete prediction,
move dcms that are missing, convert them to nii """
#%% 
# In[Import]
import os
from os.path import join, split
from pathlib import Path
import pickle
import json
import shutil
from time import time
from utilities.basic import list_subdir
#%% 
# In[tables directories]
script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parent
disk_dir = "F:\\"
sif_dir = "Y:\\"
dst_data_dir = f"{disk_dir}/CoBra/Data/dcm"
data_dir = join(base_dir, 'data')
table_dir = join(data_dir, 'tables')
update_downloaded_files = False
pred_dir = join(disk_dir, "CoBra\\Data\\volume_longitudinal_nii\\prediction")
input_dir = join(disk_dir, "CoBra\\Data\\volume_longitudinal_nii\\input\\nii_files")
data_long_dir = join(data_dir, 't1_longitudinal')
results_dir = join(data_long_dir, 'results')

with open(join(table_dir, 'sif_series_directories.pkl'), 'rb') as f:
    sif_volume_dir_dic = pickle.load(f)
with open(join(table_dir, "disk_series_directories.json"), 'rb') as f:
    dir_dic = json.load(f)

#%% 
with open(join(table_dir, 'sif_series_directories.pkl'), 'rb') as f:
    sif_volume_dir_dic = pickle.load(f)
with open(join(data_long_dir, "sids_long_new.pkl"), 'rb') as f:
    long_sids_ls = pickle.load(f)
with open(join(table_dir, "disk_series_directories.json"), 'rb') as f:
    dir_dic = json.load(f)
with open(join(table_dir, 'newIDs_dic.pkl'), 'rb') as f:
    id_dic = pickle.load(f)
with open(join(results_dir, 'missing_dcms_sids.pkl'), 'rb') as f:
        missing_dcms_sids= pickle.load(f)


missing_dcms_nids = [id_dic[sid] for sid in missing_dcms_sids]
print('remove:' ,missing_dcms_nids)

def remove_pred_files(missing_dcms_nids):
    pred_nii_files = [(join(pred_dir, nid+'_seg.nii.gz'), join(pred_dir, nid+'_1mm_seg.nii.gz'),join(pred_dir, nid+'_1mm.nii.gz')) \
                    for nid in missing_dcms_nids]
    for pred_tuple in pred_nii_files:
        for pred_file in pred_tuple:
            if os.path.exists(pred_file):
                os.remove(pred_file)
def remove_input_files(missing_dcms_nids):
    input_nii_files = [join(input_dir, nid+'.nii.gz') for nid in missing_dcms_nids]
    for input_file in input_nii_files:
        if os.path.exists(input_file):
            os.remove(input_file)
def move_missing_dcms(missing_dcms_sids):
    print('Move missing files')
    for missing_dcms_sid in missing_dcms_sids:
        print('move from ',join(sif_dir, sif_volume_dir_dic[missing_dcms_sid]),'to', dir_dic[missing_dcms_sid])
        for dcm_file in list_subdir(join(sif_dir, sif_volume_dir_dic[missing_dcms_sid])):
            dcm_id = split(dcm_file)[1]
            tgt_file = join(disk_dir, dir_dic[missing_dcms_sid], dcm_id)
            # tgt_file = join(disk_dir, 'test', dcm_id)
            if not os.path.exists(tgt_file):
                shutil.copy(dcm_file, tgt_file)
        print(missing_dcms_sid, 'copied')
#move_missing_dcms(missing_dcms_sids)