#%%
import os
from os.path import join, split
import shutil
import pandas as pd
import pickle as pkl
import json
#%%
csv_path = "C:\\Users\\kiril\\OneDrive - University of Copenhagen\\Cobra\\CMB_paper"
dist_path = "C:\\Users\\guest_acc\\CoBra\\data"
with open("G:\\CoBra\\Data\\metadata\\tables\\newIDs_dic.pkl", 'rb') as f:
    newIDs_dic = pkl.load(f)
with open("G:\\CoBra\\Data\\metadata\\tables\\disk_series_directories.json", 'rb') as f:
    dir_dic = json.load(f)
segmented_path = "C:\\Users\\guest_acc\\CoBra\\data\\prediction-new"
segmented_ids = [f[:6] for f in os.listdir(segmented_path) if f.endswith("1mm.nii.gz")]
print(dir_dic)
#%%
df = pd.read_csv(join(csv_path,"included_nii_v5_names.csv"))
for seg_id in segmented_ids:
    old_path = split(split(row.old_path)[0])[0]
    src_path = join('Y:\\', old_path, 'DOC')
    mri_name = row.new_name
    dst_path = join("G:\\CoBra\\Data\\swi_nii\\cmb_study\\reports", 
                    mri_name[:-7])
    try:
        shutil.copytree(src_path, dst_path)
    except: pass