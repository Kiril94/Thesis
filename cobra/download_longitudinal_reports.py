#%%
import os
from os.path import join, split
import shutil
import pandas as pd
import pickle as pkl
import json
from utilities.basic import get_part_of_path
import glob
#%%
dst_dir = "C:\\Users\\guest_acc\\CoBra\\data"
with open("G:\\CoBra\\Data\\metadata\\tables\\newIDs_dic.pkl", 'rb') as f:
    newIDs_dic = pkl.load(f)
newIDs_dic = {v: k for k, v in newIDs_dic.items()}
with open("G:\\CoBra\\Data\\metadata\\tables\\disk_series_directories.json", 'rb') as f:
    dir_dic = json.load(f)
segmented_path = "C:\\Users\\guest_acc\\CoBra\\data\\prediction-new"
segmented_ids = [f[:6] for f in os.listdir(segmented_path) if f.endswith("1mm.nii.gz")]
for k,v in dir_dic.items():
    dir_dic[k] = join('G:\\', get_part_of_path(v,1,6))
df_dirs = pd.read_csv("G:\\CoBra\\Data\\metadata\\tables\\series_directories_sif.csv")
sif_dir_dic = dict(zip(df_dirs.SeriesInstanceUID,df_dirs.Directory))
for k,v in sif_dir_dic.items():
    sif_dir_dic[k] = join('Y:\\', get_part_of_path(v, 0,3))

#%%
print(sif_dir_dic[list(sif_dir_dic.keys())[0]], list(sif_dir_dic.keys())[0])
#%%

for i, newid in enumerate(segmented_ids):
    print('.', end='')
    if i % 100 == 0:
        print('|', end='')
    sid = newIDs_dic[newid]
    src_path = dir_dic[sid]
    docs_ls = [f for f in glob.glob(join(src_path,'DOC/*.pdf'))]
    if len(docs_ls)==0:
        src_path = sif_dir_dic[sid]
        print(src_path)
        docs_ls = [f for f in glob.glob(join(src_path,'DOC/*/*.pdf'))]
    if len(docs_ls)==0:
        continue
    elif len(docs_ls)==1:
        src_path = docs_ls[0]
        dst_path = join(dst_dir, 'reports', newid+'.pdf')
        if not os.path.exists(dst_path):
            shutil.move(src_path, dst_path)
    else:
        for i, doc_file in enumerate(docs_ls):
            dst_path = join(dst_dir, 'reports', f"{newid}_{i}.pdf")
            if not os.path.exists(dst_path):
                shutil.move(doc_file, dst_path)

