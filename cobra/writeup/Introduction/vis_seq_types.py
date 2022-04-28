#%%
import os, sys
import string
import pickle
from os.path import split, join
import gzip
base_dir = split(split(os.getcwd())[0])[0]
if base_dir not in sys.path:
    sys.path.append(base_dir)
import pandas as pd
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import json
import pylibjpeg
#import proplot as pplt
import matplotlib.image as mpimg
import matplotlib as mpl
#%%
table_dir = f"{base_dir}/data/tables"
with open(join(table_dir, 'scan_tables','scan_after_sq_pred_dst_nos_date.pkl'), 'rb') as f:
    dfc = pickle.load(f)
#with open(join(table_dir, 'sif_series_directories.pkl'), 'rb') as f:
#    dir_dic = pickle.load(f)
with open(join(table_dir, 'disk_series_directories.json'), 'rb') as f:
    dir_dic = json.load(f)
#with open("F:\\CoBra\\Data\\dcm\\volume_log.txt", 'r') as f:
#    a = f.read()
disk_dir = "F:\\CoBra\\Data\\dcm"
#print(a)
dw_sids = np.loadtxt("F:\\CoBra\\Data\\dcm\\volume_log.txt", dtype=str)
dfc = dfc[dfc.SeriesInstanceUID.isin(dw_sids)]
#dwnld_vol_log = 

#%%
examples_dir = "C:\\Users\\kiril\\Thesis\\CoBra\\cobra\\writeup\\Introduction\\example_mri_seqs"
im_dwi = join(examples_dir, 'dwi_example2.png')
im_swi = join(examples_dir, 'swi_example.png')
im_flair = join(examples_dir, 'flair_example.png')
im_t1 = join(examples_dir, 't1_example.png')
im_t2 = join(examples_dir, 't2_example.png')

fig, ax = plt.subplots(1,5, constrained_layout=True, figsize=(10,3))
ax[0].imshow(mpimg.imread(im_t1),aspect='auto')
ax[1].imshow(mpimg.imread(im_t2),aspect='auto')
ax[2].imshow(mpimg.imread(im_flair),aspect='auto')
ax[3].imshow(mpimg.imread(im_swi),aspect='auto')
ax[4].imshow(mpimg.imread(im_dwi),aspect='auto')

for i, a in enumerate(ax):
    a.text(0, 1.05, string.ascii_lowercase[i], transform=a.transAxes, 
            size=20, weight='bold')
    a.axis('off')
fig.savefig(join(examples_dir, 'combined.png'),bbox_inches='tight', dpi=1000)
    
#mpl.pyplot.subplots_adjust(bottom=0,top=1, wspace=.01)
#%%
dfc.groupby('PatientID').Sequence.unique().iloc[10:100].head(30)

#%%
df_all = dfc[dfc.PatientID.isin(['008cc422437a23830454aeae13a0eb13'])]
df_all
#df_all[df_all.Sequence=='flair']
#%%
flair_slice = 20
flair_num = 0
df_flair = df_all[df_all.Sequence=='dwi']
dir_flair = dir_dic[df_flair.iloc[0,flair_num]]
data = pydicom.dcmread(join(disk_dir, dir_flair, os.listdir(dir_flair)[flair_slice]))
arr = data.pixel_array
#swi_dir = dir_dic[df_swi.iloc[50,0]]
#t1s = df1p[df1p.Sequence=='t1'].iloc[0,0]
#dataset = pydicom.dcmread(join(disk_dir, swi_dir, os.listdir(swi_dir)[1]))
#swi_dir
#arr = dataset.pixel_array
#dir_dic['458b4cb19d72f3b92bb738ca719ad1cc']
fig, ax = plt.subplots(1,4, figsize=(10,10))
ax[0].imshow(arr,cmap='gray')
#%%

