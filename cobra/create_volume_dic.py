"""Create a dictionary (in parallel) with
all the downloaded files as keys (niis or dcms) and the correpsonding directory
on the disk as value."""

import os
from os.path import join, split
from pathlib import Path
import json
import time
import multiprocessing


script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parent
disk_dir = "F:"
dst_data_dir = f"{disk_dir}/CoBra/Data/dcm"
data_dir = join(base_dir, 'data')
table_dir = join(data_dir, 'tables')
disk_dcm_dir = 'F:\\CoBra\\Data\\dcm'
disk_nii_dir = 'F:\\CoBra\\Data\\nii'


def get_dic(vol_dir_dic, month_dir, filetype='dcm'):
    pat_dirs = [join(month_dir, f) for f in os.listdir(month_dir)]
    for pat_dir in pat_dirs:
        if filetype=='dcm':
            vols = [f for f in os.listdir(pat_dir) if f!='DOC' \
                                and len(os.listdir(join(pat_dir, f)))>0]
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

def main(filetype):
    manager = multiprocessing.Manager()
    shared_vol_dir_dic = manager.dict()
    
    if filetype=='dcm':
        base_data_dir = disk_dcm_dir
    elif filetype=='nii':
        base_data_dir = disk_nii_dir    
    else:
        print("Select nii or dcm as filetype")
    
    months_dirs = [join(base_data_dir, f) for f in os.listdir(base_data_dir) if f.startswith('2019')
                        or f.startswith('pos')] 
    procs = []
    for month_dir in months_dirs:
        p = multiprocessing.Process(
            target=get_dic, args=[shared_vol_dir_dic, month_dir, filetype])
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    if filetype=='nii':
        with open(join(table_dir, 'disk_series_directories_niis.json'), 'w') as fp:
                json.dump(shared_vol_dir_dic.copy(), fp)
    elif filetype=='dcm':
        with open(join(table_dir, 'disk_series_directories.json'), 'w') as fp:
            json.dump(shared_vol_dir_dic.copy(), fp)
    else:
        print("Select nii or dcm as filetype")
    
if __name__ == '__main__':
    filetype = 'dcm'
    start = time.time()
    main(filetype)
    print(time.time()-start)

    
   
    
    
