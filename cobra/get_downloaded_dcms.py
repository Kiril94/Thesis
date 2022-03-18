from os.path import join, split
import os

disk_dcm_dir = "F:/Cobra/Data"



def get_dirs():
    months_dirs = [join(disk_dcm_dir, f) for f in os.listdir(disk_dcm_dir) if f.startswith('2019')
                        or f.startswith('pos')] 
    for month_dir in months_dirs[:1]:
        pat_dirs = [join(month_dir, f) for f in os.listdir(month_dir)]
    print(len(pat_dirs))