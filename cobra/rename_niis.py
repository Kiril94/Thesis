"""Creating new ids for converted niis. Ranging from 000000 to num_scans
The result is stored in tables_dir = join(base_dir, 'data', 'tables','newIDs_dic.pkl') """

from os.path import join, split
import os
import time
from utilities.basic import list_subdir
import pickle
from pathlib import Path
import shutil
script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parent
tables_dir = join(base_dir, 'data', 'tables')
disk_data_dir = join("F:\\", 'CoBra', 'Data')
nii_input_dir = join(disk_data_dir, 'volume_longitudinal_nii', 'input')
cases_dir = join(nii_input_dir, 'cases')
con_dir = join(nii_input_dir, 'potential_controls')
with open(join(tables_dir, 'newIDs_dic.pkl'), 'rb') as f:
    id_dic = pickle.load(f)


def move_files(dir_):
    start = time.time()
    for i, pat_dir in enumerate(list_subdir(dir_)):
        if i%100==0:
            print('.')
        for file in list_subdir(pat_dir):
            sid = split(file[:-7])[1]
            try:
                newID = id_dic[sid]
            except:
                print(f'Problem with file {file}')
                if sid.endswith('_0'):
                    os.remove(file)
                    continue
                elif sid.endswith('_1'):
                    sid = sid[:-2]
                    newID = id_dic[sid]
                else:
                    os.remove(file)
                    continue
            new_file = join(nii_input_dir, 'nii_files', newID+'.nii.gz')
            shutil.move(file, new_file)
    print(f"{time.time()-start:.2f}", ' s')

def remove_empty_folders(dir_):
    for pat_dir in list_subdir(dir_):
        if len(os.listdir(pat_dir))==0:
            shutil.rmtree(pat_dir)
#remove_empty_folders(con_dir)
#move_files(con_dir)
#remove_empty_folders(con_dir)
#print(id_dic["d145940e00926dd502aa97c28e76c9635"])
