# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 15:50:50 2021

@author: klein
"""
#import xgboost
import os
from pathlib import Path
import pandas as pd
#from utilss import stats
from utilss import utils
from vis import vis
from utilss import mri_stats
from utilss.basic import DotDict
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing as pp


# In[tables directories]
script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parent
fig_dir = f"{base_dir}/figs/basic_stats"
table_dir = f"{base_dir}/tables"
fig_dir = f"{base_dir}/figs"

# In[Define useful keys]
TE_k = 'EchoTime'
TR_k = 'RepetitionTime'
TI_k = 'InversionTime'
FA_k = 'FlipAngle'
SD_k = 'SeriesDescription'
PID_k = 'PatientID'
SID_k = 'SeriesInstanceUID'
SS_k = 'ScanningSequence'
SV_k = 'SequenceVariant'
SO_k = 'ScanOptions'
ETL_k = 'EchoTrainLength'

# In[load all csv]
rel_cols = [SID_k, SD_k, TE_k, TR_k, FA_k, TI_k, ETL_k, SS_k, SV_k, PID_k]
table_all_dir = f"{table_dir}/neg_pos.csv"  
df_all = utils.load_scan_csv(table_all_dir)[rel_cols]

# In[Select only relevant columns]
print(f"all elements {len(df_all)}")
df_all = df_all[rel_cols].dropna(subset=[SID_k, PID_k, SS_k, SV_k, TR_k])
print(f"after dropping nans {len(df_all)}")


# In[Get masks for the different series descriptions]
mask_dict, tag_dict = mri_stats.get_masks_dict(df_all)

# In[Add sequence column and set it to one of the relevant values]
sq = 'Sequence'
rel_keys = ['t1', 't1gd', 't2', 't2gd', 't2s', 'swi', 'flair','none_nid', 
            'gd', 'dwi',]
rel_masks = [mask_dict[key] for key in rel_keys] 
df_all[sq] = "other"
for mask, key in zip(rel_masks, rel_keys):
    df_all[sq][mask] = key
df_all[sq][mask_dict.t1gd] = "t1"
df_all[sq][mask_dict.t2gd] = "t2"
df_all[sq][mask_dict.gd] = "none_nid"

# In[Count number of volumes for every sequence]
seq_count = df_all[sq].value_counts()
print(seq_count)

# In[visualize number of volumes sequences]
vis.bar_plot(seq_count.keys(), seq_count.values, figsize=(13,6), xlabel='Sequence',
             xtickparams_ls=16, save_plot=True, title='All Patients',
             figname=f"{fig_dir}/sequence_pred/sequences_count.png")


# In[Turn ScanningSequence into multi-hot encoded]
s = df_all[SS_k].explode()
columns_list = list(df_all.columns)
columns_list.remove(SS_k)
df_all = df_all[columns_list].join(pd.crosstab(s.index, s))
del s
# In[Turn SequenceVariant into multi-hot encoded]
s = df_all[SV_k].explode()
columns_list = list(df_all.columns)
columns_list.remove(SV_k)
df_all = df_all[columns_list].join(pd.crosstab(s.index, s))
del s
# In[Fraction of missing values in each column]
print(df_all.isna().mean(axis=0))
# In[Lets set missing inversion times, Flip Angles and Echo times to 0]
df_all[df_all[TI_k].isna()] = 0
df_all[df_all[FA_k].isna()] = 0
df_all[df_all[TE_k].isna()] = 0
# In[Now we can show the histograms, except non numeric values]
columns_list = list(df_all.columns)
sparse_columns = ['EP', 'GR', 'IR', 'RM', 'SE', 'DE', 'MP', 'MTC', 
                  'OSP', 'SK', 'SP', 'SS', 'TOF']
rmv_list = [SID_k, PID_k] + sparse_columns

for itm in rmv_list:
    columns_list.remove(itm)
fig, ax = plt.subplots(figsize=(10,10))
ax = df_all.hist(column=columns_list, figsize=(10,10), ax=ax, log=True)
fig.tight_layout()
fig.savefig(f"{fig_dir}/sequence_pred/X_distr.png")
# In[Show the binary columns]
bin_counts = df_all[sparse_columns].sum(axis=0)
vis.bar_plot(bin_counts.keys(), bin_counts.values, 
             figsize=(15,6), logscale=True, 
             figname=f"{fig_dir}/sequence_pred/X_distr_bin.png")
# In[We can drop the TOF, EP, DE and MTC columns]
df_all = df_all.drop(['TOF', 'EP', 'DE', 'MTC'], axis=1)


# In[Before encoding we have to split up the sequences we want to predict (test set)]
df_test = df_all[df_all.Sequence == 'none_nid']
df_train = df_all[df_all.Sequence != 'none_nid']
del df_all

# In[]
print(df_train.Sequence)
# In[One hot encode target]
lb = pp.LabelBinarizer()
lb.fit(df_train.Sequence)
y = lb.transform(df_train.Sequence)
print(y.shape)

# In[Now we separate the patient ID and the SeriesInstance UID]
#From now on we should not change the order or remove any values,
# otherwise the ids wont match afterwards
df_ids = df_all[[PID_k, SID_k]]
df_all = df_all.drop([PID_k, SID_k, 'Sequence', SD_k], axis=1) 
# Drop also the sequence and series description column

# In[Drop series description]
df_all = df_all.drop([SD_k], axis=1)
#no remaining nan values
print(df_all.isnull().sum())

# In[Min max scale non bin columns]

num_cols = df_all.select_dtypes(include="float64").columns
X = df_all.copy()
for col in num_cols:
  X[col] = (df_all[col] - df_all[col].min())/(df_all[col].max()-df_all[col].min()) 

# In[Train val separation]
test_label = lb.transform(np.array(['none_nid']))








# In[Turn sparse columns into sparse arrays]
columns_list = list(df_all.columns)
sparse_columns = ['EP', 'GR', 'IR', 'RM', 'SE', 'DE', 'MP', 'MTC', 
                  'OSP', 'SK', 'SP', 'SS', 'TOF']
for item in sparse_columns:
    columns_list.remove(item)

df_all = utils.convert_to_sparse_pandas(
    df_all, columns_list)
print(df_all.dtypes)




# In[Count number of relevant patients in 2019]
print(len(df_2019[mask_dict.t1 | mask_dict.t2 | mask_dict.t2s | mask_dict.t1gd \
        | mask_dict.t2gd | mask_dict.gd | mask_dict.swi | mask_dict.flair \
        | mask_dict.dwi][PID_k].unique()))
print(len(df_2019[PID_k].unique()))



# In[visualize sequences scatter]
fig, ax = plt.subplots(2,2,figsize=(10,10))
ax = ax.flatten()
sns.scatterplot(x=TE_k, y=TR_k, data=df_all, hue='Sequence',ax=ax[0])
sns.scatterplot(x=TE_k, y=IR_k, legend=None,hue='Sequence',
                data=df_all,
                ax=ax[1])
sns.scatterplot(x=IR_k, y=TR_k, legend=None,hue='Sequence',
                data=df_all,
                ax=ax[2])
sns.scatterplot(x=IR_k, y=FA_k, legend=None,hue='Sequence',
                data=df_all,
                ax=ax[3])
fig.suptitle('All Sequences', fontsize=20)
fig.tight_layout()
fig.savefig(f"{fig_dir}/sequence_pred/scatter_for_all.png")

# In[tets]
print(df_all.keys())

# In[https://towardsdatascience.com/beginners-guide-to-xgboost-for-classification-problems-50f75aac5390]