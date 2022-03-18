import os, sys
from os.path import join, split
ROOT_DIR = os.path.abspath(os.curdir)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
    
import get_downloaded_dcms

disk_dcm_dir = "F:/Cobra/Data"

months_dirs = [join(disk_dcm_dir, f) for f in os.listdir(disk_dcm_dir) if f.startswith('2019')
                        or f.startswith('pos')] 
for month_dir in months_dirs[:1]:
    pat_dirs = [join(month_dir, f) for f in os.listdir(month_dir)]
print(len(pat_dirs))