#%%
import os, sys
import pickle
from os.path import split, join
base_dir = split(os.getcwd())[0]
if base_dir not in sys.path:
    sys.path.append(base_dir)
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
#%%
# In[dirs and load df]
script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parent.parent
table_dir = f"{base_dir}/data/tables/scan_tables"
with open(join(table_dir, 'scan_after_sq_pred.pkl'), 'rb') as f:
    df = pickle.load(f)
#%%
print(len(df[df.Sequence=='flair']))
print(len(df[df.Sequence=='swi']))
df[df.Sequence=='t1'].PatientID.nunique()
#%%
flair_pat = set(df[df.Sequence=='flair'].PatientID.unique())
swi_pat = set(df[df.Sequence=='swi'].PatientID.unique())
flair_swi_pat_n = len(flair_pat.intersection(swi_pat))

#%%
sns.set_theme('paper')
plt.rcParams["figure.dpi"] = 300
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

arr = np.array([[len(flair_pat), flair_swi_pat_n],
                [flair_swi_pat_n, len(swi_pat)]])
ax = sns.heatmap(arr, annot=True, fmt=".0f", ax=ax, )
ax.set_yticklabels(['FLAIR','SWI',])
ax.set_xticklabels(['FLAIR','SWI',])
ax.set_title('# Patients')
fig.savefig('cm.png')
print(len(df[df.Sequence=='flair']))
print(len(df[df.Sequence=='swi']))
#%%
