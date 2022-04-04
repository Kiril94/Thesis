#%%
import os, sys
import pickle
from os.path import split, join

from sklearn.metrics import confusion_matrix
base_dir = split(os.getcwd())[0]
if base_dir not in sys.path:
    sys.path.append(base_dir)
from pathlib import Path
import seaborn as sns
from utilities import utils
#%%
# In[dirs and load df]
script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parent.parent
table_dir = f"{base_dir}/data/tables/scan_tables"

with open(join(table_dir, 'scan_after_sq_pred.pkl'), 'rb') as f:
    df = pickle.load(f)
df1 = df[df.Sequence=='t1']
def print_pos_neg(df):
    print('neg',df[df.Positive==0].PatientID.nunique())
    print('pos', df[df.Positive==1].PatientID.nunique())

print('initial length:', len(df1))
print_pos_neg(df1)

#%%
# Remove scans with missing dates for those who don't have 2019 tag
df1_0 = df1[~((df1['2019']==0) & (df1.InstanceCreationDate.isna()))]
print('removing missing dates from 2020/2021:', len(df1_0))
print_pos_neg(df1_0)

#%%
# In[Get number of slices if missing]
print("We need the number of slices for scans that have as MRAcquisitionType either nan or 3D")
import importlib
importlib.reload(utils)
df1_3dnone = df1_0[(df1_0.MRAcquisitionType=='3D') | (df1_0.MRAcquisitionType.isna())]
df1_3dnone_nos_miss = df1_3dnone[df1_3dnone.NumberOfSlices.isna()]
print(len(df1_3dnone_nos_miss))
n_slices_dic = utils.save_number_of_slices_to_txt(df1_3dnone_nos_miss, 'nos.txt', 
                            sif_path='Y://', disk_path='F:')
#%%
# In[Save dict]
with open('nos.pkl', 'wb') as f:
    pickle.dump(n_slices_dic, f)
#%%
# In[Keep scan if 3D or if missing if number of slices>64]
df1_0[df1_0.MRAcquisitionType.isna()].NumberOfSlices.hist()

#%%
df1_0.NumberOfSlices.isna().mean()