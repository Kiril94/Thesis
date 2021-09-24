# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 15:50:50 2021

@author: klein
"""
import xgboost as xgb
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import scikitplot as skplot

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
    df_all[sq] = np.where((mask), key, df_all[sq])
df_all[sq] = np.where((mask_dict.t1gd), "t1", df_all[sq])
df_all[sq] = np.where((mask_dict.t2gd), "t2", df_all[sq])
df_all[sq] = np.where((mask_dict.gd), "none_nid", df_all[sq])
print(df_all[sq])

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
df_all[TI_k] = np.where((df_all[TI_k].isna()), 0, df_all[TI_k])
df_all[FA_k] = np.where((df_all[FA_k].isna()), 0, df_all[FA_k])
df_all[TE_k] = np.where((df_all[TE_k].isna()), 0, df_all[TE_k])

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
try:
    df_all = df_all.drop(['TOF', 'EP', 'DE', 'MTC'], axis=1)
except: 
    print("those columns are not present")
# In[Before encoding we have to split up the sequences we want to predict (test set)]
df_test = df_all[df_all.Sequence == 'none_nid']
df_train = df_all[df_all.Sequence != 'none_nid']
del df_all

# In[Integer encode labels]
target_labels = df_train.Sequence.unique()
target_ids = np.arange(len(target_labels))
target_dict = dict(zip(target_labels, target_ids))
y = df_train[sq].map(target_dict)
print(y)
print(target_dict)


# In[Now we separate the patient ID and the SeriesInstance UID]
#From now on we should not change the order or remove any values,
# otherwise the ids wont match afterwards
df_train_ids = df_train[[PID_k, SID_k]]
df_test_ids = df_test[[PID_k, SID_k]]
try:
    df_train = df_train.drop([PID_k, SID_k, 'Sequence', SD_k], axis=1)
    df_test = df_test.drop([PID_k, SID_k, 'Sequence', SD_k], axis=1)
except:
    print("those columns are not present")

# In[Min max scale non bin columns]

num_cols = df_train.select_dtypes(include="float64").columns
X = df_train.copy()
X_test = df_test.copy()
for col in num_cols:
  X[col] = (df_train[col] - df_train[col].min())/ \
      (df_train[col].max()-df_train[col].min()) 
  X_test[col] = (df_test[col] - df_test[col].min())/ \
      (df_test[col].max()-df_test[col].min()) 


# In[Split train and val]
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42)

# In[Initialize and train]
xgb_cl = xgb.XGBClassifier() 
xgb_cl.fit(X_train, y_train)

# In[Predict]
pred_prob_val = xgb_cl.predict_proba(X_val)

# Score
#accuracy_score(y_test, preds)

# In[Plot roc curve]
skplt.metrics.plot_roc_curve(y_val, pred_prob_val, figsize=(9,8),
                             text_fontsize=12,
                             title="Sequence Prediction - ROC Curves")

# In[]
pred_val = np.copy(pred_prob_val)
pred_val[pred_val>.9] = 1
pred_val[pred_val<=.9] = 0
pred_val = np.argmax(pred_val, axis=1)

# In[]
ax = skplot.metrics.plot_confusion_matrix(y_val, pred_val, normalize=True,
                                     figsize=(8,8))

# In[]

print(y_val)
print(target_dict)









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


# In[https://towardsdatascience.com/beginners-guide-to-xgboost-for-classification-problems-50f75aac5390]