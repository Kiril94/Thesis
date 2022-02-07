import os
from os.path import join, split
from pathlib import Path
import json
import time

script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parent
disk_dir = "F:"
dst_data_dir = f"{disk_dir}/CoBra/Data/dcm"
data_dir = join(base_dir, 'data')
table_dir = join(data_dir, 'tables')
disk_dcm_dir='F:\\CoBra\\Data\\dcm'
disk_nii_dir = 'F:\\CoBra\\Data\\nii'


def get_dic(base_data_dir, filetype='dcm'):
    vol_dir_dic = dict()
    months_dirs = [join(base_data_dir, f) for f in os.listdir(base_data_dir) if f.startswith('2019')
                        or f.startswith('pos')]
    for month_dir in months_dirs:
        pat_dirs = [join(month_dir, f) for f in os.listdir(month_dir)]
        for pat_dir in pat_dirs:
            if filetype=='dcm':
                vols = [f for f in os.listdir(pat_dir) if f!='DOC']
                if len(vols)>0:
                    for vol in vols:
                        vol_dir_dic[vol] = join(pat_dir, vol)        
            elif filetype=='nii':
                vols = [f for f in os.listdir(pat_dir) if f.endswith('.nii')]
                single_vols = [f for f in vols if '_' not in f]
                mul_vols = [f for f in vols if '_00002' in f]
                single_vols_ids = [split(f)[1][:-4] for f in single_vols]
                mul_vols_ids = [str(split(f)[1]).split('_')[0] for f in mul_vols]
                all_vols = single_vols + mul_vols
                all_vols_ids = mul_vols_ids + single_vols_ids
                if len(all_vols)>0:
                    for id, dir_ in zip(all_vols_ids, all_vols):
                        vol_dir_dic[id] = join(pat_dir, dir_)
            else:
                print("Select nii or dcm as filetype")
    return vol_dir_dic


filetype = 'nii'
start = time.time()
if filetype=='nii':
    vol_nii_dic = get_dic(disk_nii_dir, filetype=filetype)
    with open(join(table_dir, 'disk_series_directories_niis.json'), 'w') as fp:
        json.dump(vol_nii_dic, fp)
elif filetype=='dcm':
    vol_dcm_dic = get_dic(disk_nii_dir, filetype=filetype)
    with open(join(table_dir, 'disk_series_directories.json'), 'w') as fp:
        json.dump(vol_dcm_dic, fp)
else:
     print("Select nii or dcm as filetype")
print(time.time()-start)