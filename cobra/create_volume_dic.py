import os
from os.path import join
from pathlib import Path
import json

script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parent
disk_dir = "F:"
dst_data_dir = f"{disk_dir}/CoBra/Data/dcm"
data_dir = join(base_dir, 'data')
table_dir = join(data_dir, 'tables')
disk_dcm_dir='F:\\CoBra\\Data\\dcm'
vol_dir_dic = dict()

months_dirs = [join(disk_dcm_dir, f) for f in os.listdir(disk_dcm_dir) if f.startswith('2019')
                    or f.startswith('pos')]

for month_dir in months_dirs:
    pat_dirs = [join(month_dir, f) for f in os.listdir(month_dir)]
    for pat_dir in pat_dirs:
        vols = [f for f in os.listdir(pat_dir) if f!='DOC']
        if len(vols)>0:
            for vol in vols:
                vol_dir_dic[vol] = join(pat_dir, vol)        
with open(join(table_dir, 'disk_series_directories.json'), 'w') as fp:
    json.dump(vol_dir_dic, fp)
