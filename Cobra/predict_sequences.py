# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 15:50:50 2021

@author: klein
"""
import xgboost as xgb
import os
from pathlib import Path
import pandas as pd
from utilss import utils
from utilss import classification as clss
from vis import vis
from stats_tools import vis as svis
from utilss import mri_stats
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
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
df_train_ids = df_train[[PID_k, SID_k,sq]]  #also contains sequences
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
# In[Plot roc curve]
vis.plot_decorator(skplot.metrics.plot_roc_curve, args=[y_val, pred_prob_val,], 
                   kwargs={'figsize':(9,8),'text_fontsize':14.5,
                           'title':"Sequence Prediction - ROC Curves"},
                   figname=f"{fig_dir}/sequence_pred/ROC_curves.png")
# In[Test FPR for different thresholds]
thresholds = np.linspace(.8,.999, 200)
fprs = []
for i, th in enumerate(thresholds):
    pred_val = clss.prob_to_class(pred_prob_val, th, 0)
    cm = confusion_matrix(y_val, pred_val)
    fprs.append(clss.fpr_multi(cm))
fprs = np.array(fprs)

# In[Plot FPR for different thresholds]
final_th = 0.92
fig, ax = plt.subplots(figsize=(9,5))
for i in range(6):
    ax.plot(thresholds, fprs[:,1+i], label=list(target_dict.keys())[i+1])
ax.axvline(final_th, color='red', linestyle='--',label=f'final cutoff={final_th}')
ax.legend(fontsize=16.5, facecolor='white')
ax.set_xlabel('Cutoff', fontsize=20)
ax.set_ylabel('FPR', fontsize=20)
fig.tight_layout()
fig.savefig(f"{fig_dir}/sequence_pred/fpr_cutoff.png", dpi=80)
# In[Plot confusion matrix]
pred_val = clss.prob_to_class(pred_prob_val, final_th, 0) # np.argmax(pred_val, axis=1)
cm = confusion_matrix(y_val, pred_val)

args = [y_val, pred_val]
kwargs = {'normalize':True, 'text_fontsize':16,'title_fontsize':18, }
vis.plot_decorator(skplot.metrics.plot_confusion_matrix, args, kwargs, 
                   set_xticks=True, xticks=np.arange(7), xtick_labels=target_dict.keys(),
                   set_yticks=True, yticks=np.arange(7), ytick_labels=target_dict.keys(),
                   save=True, figname=f"{fig_dir}/sequence_pred/confusion_matrix_norm.png")
# In[make prediction for the test set]
pred_prob_test = xgb_cl.predict_proba(X_test)
pred_test = clss.prob_to_class(pred_prob_test, final_th, 0)
vis.bar_plot(target_dict.keys(), np.unique(pred_test, return_counts=True)[1],
             xlabel='Sequence', title='Predicted sequences')

# In[Create and save dataframe with predictions]
df_test[sq] = pred_test
df_test = pd.concat([df_test, df_test_ids], axis=1)
df_train = pd.concat([df_train, df_train_ids], axis=1)
df_final = pd.concat([df_train, df_test])
del df_test, df_train
# In[Examine final df]
print(df_final.Sequence)






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