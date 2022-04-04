# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 15:50:50 2021

@author: klein
"""
#%%
from collections import defaultdict, deque
import xgboost as xgb
import os
import pickle
import json
from os.path import join
from pathlib import Path
import pandas as pd
from utilities import basic, mri_stats
from stats_tools import vis as svis
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import scikitplot as skplot
from sklearn.manifold import TSNE
from numpy.random import default_rng
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from utilities import bayesian_opt
import matplotlib.pylab as pylab
params = {'figure.dpi':300,
        'legend.fontsize': 18,#
        'figure.figsize': [8, 5],
         'axes.labelsize': 18,
         'axes.titlesize':18,
         'xtick.labelsize':18,
         'ytick.labelsize':18}
pylab.rcParams.update(params)
plt.style.use('ggplot')
#%%
# In[tables directories]
script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parent
table_dir = f"{base_dir}/data/tables"
fig_dir = f"{base_dir}/figs"

#%%
# In[Define useful keys]
TE_k = 'EchoTime'
TR_k = 'RepetitionTime'
TI_k = 'InversionTime'
FA_k = 'FlipAngle'
EN_k = 'EchoNumbers'
IF_k = 'ImagingFrequency'
SD_k = 'SeriesDescription'
PID_k = 'PatientID'
SID_k = 'SeriesInstanceUID'
SS_k = 'ScanningSequence'
SV_k = 'SequenceVariant'
SO_k = 'ScanOptions' # List of values like FS, PFP ,...
ETL_k = 'EchoTrainLength'
DT_k = 'DateTime'
ICD_k = 'InstanceCreationDate'
SN_k = 'SequenceName'
PSN_k = 'PulseSequenceName'
IT_k = 'ImageType'
sq = 'Sequence'
#%%
# In[load all csv]
df_init = pd.read_pickle(f"{table_dir}/scan_tables/scan_init.pkl")
# df_init = pd.read_csv(
    # join(table_dir, 'neg_pos_clean.csv'), nrows=80000)
#%%
# In[Missing values]
df_all = df_init.dropna(axis=1, 
        thresh=int(0.5*df_init.shape[0] + 1)) #keep columns with less than 50% missing
print('Remaining columns after dropping those with more than 50% missing')
print(df_all.keys())
#%%
# Take only relevant columns
rel_numeric_feat_ls = ['dBdt', 'EchoTime', 'EchoTrainLength', 
                'EchoNumbers','FlipAngle', 'ImagingFrequency', 'RepetitionTime']
rel_cont_feat_ls = ['ImageType','ScanningSequence', 'SequenceVariant', 'ScanOptions']
other_rel_cols_ls = ['SeriesInstanceUID', 'StudyInstanceUID','PatientID',
                    'SeriesDescription','Positive']
print('Select only relevant columns')
df_all = df_all[other_rel_cols_ls + rel_numeric_feat_ls + rel_cont_feat_ls]


#%%
# In[Handle nans in list columns and turn strings into lists]
print(f"all elements {len(df_all)}")
df_all[SS_k].fillna('missing_SS', inplace=True)
df_all[SV_k].fillna('missing_SV', inplace=True)
df_all[SO_k].fillna('missing_SO', inplace=True)
df_all[IT_k].fillna('missing_IT', inplace=True)
print("Number of missing values for every column")
print(df_all.isna().sum(axis=0))
# df_all[SS_k] = df_all[SS_k].map(lambda x: literal_eval(x))
# df_all[SV_k] = df_all[SV_k].map(lambda x: literal_eval(x))
# df_all[SO_k] = df_all[SO_k].map(lambda x: literal_eval(x))
#%%
# In[Get masks for the different series descriptions]
mask_dict, tag_dict = mri_stats.get_masks_dict(df_all)

#%%
# In[Add sequence column and set it to one of the relevant values]

rel_seqs_keys = ['t1', 't2', 'swi', 'flair','dwi']
rel_masks = [mask_dict[key] for key in rel_seqs_keys]
df_all[sq] = "none_nid" #By default the sequence is unknown
# first add relevant sequences
for mask, key in zip(rel_masks, rel_seqs_keys):
    df_all[sq] = np.where((mask), key, df_all[sq])
#if t2 with combination with flair, choose flair
df_all[sq] = np.where((mask_dict.t1gd), "t1", df_all[sq])
df_all[sq] = np.where((mask_dict.t2gd), "t2", df_all[sq])
df_all[sq] = np.where(
    (mask_dict.other & (~(mask_dict.t1|mask_dict.t1gd|mask_dict.t2|mask_dict.t2gd|
       mask_dict.swi|mask_dict.dwi|mask_dict.flair ))), 
    "other", df_all[sq])
# Intersections
df_all[sq] = np.where((mask_dict.t1&mask_dict.t2), "none_nid", df_all[sq])
df_all[sq] = np.where((mask_dict.t1&mask_dict.flair), "none_nid", df_all[sq])

print(df_all.Sequence)
df_all.Sequence.value_counts().plot(kind='barh')






#%% 
# print some numbers of scans
rel_seq = [ 't2', 'swi', 'flair', 'dwi']
for name, mask in mask_dict.items():    
    if name in rel_seq:
        print(name, ':',mask.sum())
print('other',(mask_dict.more|mask_dict.mip| mask_dict.bold| mask_dict.loc|mask_dict.b1calib|
    mask_dict.autosave|mask_dict.screensave|mask_dict.pd| mask_dict.angio|mask_dict.survey|
    mask_dict.cest|mask_dict.asl|mask_dict.tracew|mask_dict.stir|mask_dict.adc|
    mask_dict.pwi|mask_dict.dti|mask_dict.t2s).sum())
print('t1:',(mask_dict.t1gd | mask_dict.t1).sum())
#print("There are not enough dti sequences")
#%%
# In[Count number of volumes for every sequence]
seq_count = df_all[sq].value_counts()
print(seq_count)
#%%
# In[Plot sequence counts]
# We need this later
mpl.rcParams['figure.dpi'] = 400
plt.style.use('ggplot')
#labels = ['Not \nidentified', 'other', 'T1', 'T2', 'DWI', 'FLAIR', 'SWI', 'T2*']
g = sns.countplot(x="Sequence", hue="Positive", data=df_all, hue_order=[1,0],
    order = df_all['Sequence'].value_counts().index)
#g.set(xlabel=('Sequence Type'), ylabel=('Volume Count'), xticklabels=labels)
fig = g.get_figure()
fig.tight_layout()
fig.savefig(f"{fig_dir}/sequence_pred/volumes_sequence_count.png")
#%%
# In[PLot class counts, no pos. neg.]
#fig, ax = plt.subplots()
nice_labels_dic = {'flair':'FLAIR','t2':'T2', 'other':'OIS','t1':'T1','swi':'SWI','dwi':'DWI',
    'none_nid':'Unknown'}
mpl.rcParams['figure.dpi'] = 300
print(plt.rcParams['axes.prop_cycle'].by_key()['color'])
plt.style.use('ggplot')
fig, ax = plt.subplots()
labels = [nice_labels_dic[k] for k in df_all['Sequence'].value_counts().index]
g2 = sns.countplot(x="Sequence", data=df_all, 
    order = df_all['Sequence'].value_counts().index, color='#8EBA42',ax=ax)
#g2.set(xlabel=, ylabel=('# Scans'), xticklabels=labels,)
ax.set_xlabel('Class', fontsize=19)
ax.set_ylabel('# Scans', fontsize=19)
ax.set_xticklabels(labels)
ax.set_ylim(0,81000)
#fig2 = g2.get_figure()
fig.tight_layout()
fig.savefig(f"{fig_dir}/sequence_pred_new/volumes_sequence_count.png")
print(df_all.Sequence.value_counts())
print(np.sum(df_all.Sequence.value_counts())-df_all.Sequence.value_counts()['none_nid'])
#%%
# In[plot sequence counts grouped by patients]
# we might need this later
pos_mask = df_all.Positive==1
pos_pat_count_seq = df_all[pos_mask].groupby(by='Sequence')\
    .PatientID.nunique().sort_values(ascending=False)
neg_pat_count_seq = df_all[~pos_mask].groupby(by='Sequence')\
    .PatientID.nunique().sort_values(ascending=False)
n_labels = ['Other', 'Flair', 'T2', 'Unknown', 'DWI', 'T1', 'SWI', ]
fig, ax = svis.bar(n_labels, neg_pat_count_seq.values, 
                   label='neg', color=svis.Color_palette(0)[1])

svis.bar(n_labels,pos_pat_count_seq.values, fig=fig, ax=ax,
         bottom=neg_pat_count_seq.values, label='pos',
         color=svis.Color_palette(0)[0], 
         kwargs={'lgd':True, 'xlabel':'Sequence Type','title':'All Patients',
                 'ylabel':'Patient Count'},
         save=True, 
         figname=f"{fig_dir}/sequence_pred_new/seq_patient_count.png")




#%%
# In[mask and minimum number of present entries]
train_mask = df_all.Sequence!='none_nid'
print(df_all.keys())
print('Remove binary feature if it is present in less than 1% of training samples')
min_pos = int(train_mask.sum()*0.01)
print(min_pos)
#%%
# In[Turn ScanningSequence into multi-hot encoded]
# Drop features with less than 1000 entries
s = df_all[SS_k].explode()
columns_list = list(df_all.columns)
columns_list.remove(SS_k)
ct = pd.crosstab(s.index, s)
ct_sum = ct[train_mask].sum(axis=0)
keep_cols = ct_sum.loc[(ct_sum>min_pos)].keys().tolist()
df_all = df_all.join(ct[keep_cols])
print("Unique Scanning Sequences:")
print(set(df_all.columns)-set(columns_list))
#%%
# In[Turn SequenceVariant into multi-hot encoded]
s = df_all[SV_k].explode()
columns_list = list(df_all.columns)
columns_list.remove(SV_k)
ct = pd.crosstab(s.index, s)
ct_sum = ct[train_mask].sum(axis=0)
keep_cols = ct_sum.loc[(ct_sum>min_pos)].keys().tolist()
df_all = df_all.join(ct[keep_cols])
print("Unique Sequence Variants:")
print(set(df_all.columns)-set(columns_list))

#%%
# In[Turn ScanOptions into multi-hot encoded]
# Scan options that are equivalent to 
# scanning sequence of sequence variants are combined with or
s = df_all[SO_k].explode()
columns_list = list(df_all.columns)
columns_list.remove(SO_k)
ct = pd.crosstab(s.index, s)
ct_sum = ct[train_mask].sum(axis=0)
keep_cols = ct_sum.loc[(ct_sum>min_pos)].keys().tolist()
ct = ct[keep_cols]
df_all = df_all.join(ct, rsuffix='_SO')
print("Unique Sequence Variants:")
print(set(df_all.columns)-set(columns_list))


#%%
# In[Image type to one hot]
s = df_all.ImageType.explode()
columns_list = list(df_all.columns)
columns_list.remove('ImageType')
ct = pd.crosstab(s.index, s)
ct_sum = ct[train_mask].sum(axis=0)
keep_cols = ct_sum.loc[(ct_sum>min_pos)].keys().tolist()
ct = ct[keep_cols]
df_all = df_all.join(ct, rsuffix='_IT')
print("Unique Sequence Variants:")
print(set(df_all.columns)-set(columns_list))
del s, ct

#%%
# In[Examine results]
df_all.keys()

#%% Combine duplicate columns and remove unnecessary columns
df_all.drop(columns=[SS_k, SV_k, SO_k, 'ImageType'], inplace=True, axis=1) #drop org cols
df_all.SP = (df_all.SP | df_all['SP_SO'].astype(int)).astype(int)
df_all.SE = (df_all.SE | df_all['SE_IT'].astype(int)).astype(int)
df_all.IR = (df_all.IR | df_all['IR_SO'].astype(int)\
    | df_all['IR_IT'].astype(int)).astype(int)
df_all.drop(columns=['SP_SO','IR_SO','SE_IT' ,'IR_IT',''], axis=1, inplace=True)


#%%
# In[Check new columns]
df_all.keys()
#%%

#%% 
#In [Currently not needed, too many missing values]
"""
print('Maybe not needed, first check number of missing columns in real dataframe')
psn_strings = ['tse', 'fl',  'tir', 'swi', 'ep', 'tfl', 're', 'spc', 'pc',
               'FSE','FIR','GE', 'h2', 'me', 'qD', 'ci', 'fm','de','B1']
add_psn_str = ['se', 'SE']
for psn_string in psn_strings:
    mask = df_all.SequenceName.str.startswith('*'+psn_strings[0])
    mask = mask | df_all.SequenceName.str.startswith(psn_strings[0])
    mask = mask | df_all.SequenceName.str.startswith('*q'+psn_strings[0])
    df_all[psn_string] = 0
    df_all.loc[mask, psn_string] = 1 
mask = df_all.SequenceName.str.startswith('*'+'se')
mask = mask | df_all.SequenceName.str.startswith('se')
mask = mask | df_all.SequenceName.str.startswith('*qse')
mask = mask | df_all.SequenceName.str.startswith('*'+'SE')
mask = mask | df_all.SequenceName.str.startswith('SE')
mask = mask | df_all.SequenceName.str.startswith('*qSE')
print(mask.sum())
#for psn_string in psn_strings[1:]:
#    mask = mask | df_all.SequenceName.str.startswith('*'+psn_string)
#    mask = mask | df_all.SequenceName.str.startswith(psn_string)
 #   mask = mask | df_all.SequenceName.str.startswith('*q'+psn_string)
print(df_all[~mask].SequenceName.unique())
"""




#%%
# In[Produce Pairplot]

mpl.rcParams['figure.dpi'] = 300
from IPython.display import Image
markers_dic = dict(
    zip(['t1', 't2','flair', 'dwi','swi', 'other', 'none_nid'], 
    ["d",  "d", 'd','d','d', 'd', '.']))
plot_vars = ['dBdt','EchoTime', 'RepetitionTime','FlipAngle',
    'EchoTrainLength','ImagingFrequency' ]
print('remove outliers before plotting')
Q1 = df_all[plot_vars].quantile(0.25)
Q3 = df_all[plot_vars].quantile(0.75)
IQR = Q3 - Q1
df_temp = df_all[~((df_all[plot_vars] < (Q1 - 1.5 * IQR)) |(df_all[plot_vars] > (Q3 + 1.5 * IQR))).any(axis=1)]

#%%
sns_plot = sns.pairplot(df_temp, 
    vars=plot_vars,
      diag_kind="kde", hue='Sequence', 
     hue_order=['t1', 't2', 'flair', 'dwi','swi', 'other', 'none_nid'],
    plot_kws={"s": 13, 'alpha': 1},
     markers=markers_dic, palette='Set1', corner=True,
    )
# diag_kws={'log':False}
sns_plot._legend.set_title('Class')

labels_dic = {'EchoTime':r'$t_\mathrm{echo}\,[\mathrm{ms}]$', 
    'RepetitionTime':r'$t_\mathrm{rep}\,[\mathrm{ms}]$',
    'InversionTime':'Inversion Time [ms]', 'FlipAngle':r'$\alpha_{\mathrm{flip}} \,[°]$',
    'EchoTrainLength':'ETL', '':'',
    'ImagingFrequency':r'$f_\mathrm{Im}\,[\mathrm{MHz}]$',
    'dBdt':r'$\frac{dB}{dt}\,[\mathrm{T}/\mathrm{s}]$'}

print('Limits have to be adjusted, and axis numbers too')
num_plot_vars = len(plot_vars)
ax_mat = np.tril(np.reshape(np.arange(num_plot_vars**2), (num_plot_vars, num_plot_vars)))
ax_mat = ax_mat[ax_mat!=0]
for i, ax in enumerate(sns_plot.axes.flatten()):
    if i in ax_mat:
        ax.set_xlabel(labels_dic[ax.get_xlabel()], fontsize=20)
        ax.set_ylabel(labels_dic[ax.get_ylabel()], fontsize=20)
        ax.tick_params(labelsize=15)
    # if i==6:
        # ax.set_ylim(-10,450)
    # if i in ax_mat[[1,3,6,10, 15]]:
        # ax.set_xlim(-10,510)
    # if i in ax_mat[[9,10,11,12, 13]]:
        # ax.set_xlim(-10,320)
    # if i%5==0:
        # ax.set_xlim(-10,400)
    # if i>19:
        # ax.set_ylim(-10,300)
    # if (i-1)%5==0:
        # ax.set_xlim(-400,12500)
    # if (i+1)%5==0:
        # ax.set_xlim(-10,250)
handles = sns_plot._legend_data.values()
labels = sns_plot._legend_data.keys()
new_labels = ['T1', 'T2', 'FLAIR', 'DWI','SWI', 'OIS','Unknown']
lgd_labels_dic = dict(zip(['t1','t2','flair','dwi','swi','other','none_nid'],
    new_labels))
sns_plot._legend.remove()
plt.legend(handles=handles, labels=[lgd_labels_dic[l] for l in labels], 
    loc=(-.5,3.95), ncol=1, fontsize=20)
sns_plot.savefig(f"{fig_dir}/sequence_pred_new/X_pairplot.png")
plt.clf() # Clean parirplot figure from sns 
Image(filename=f"{fig_dir}/sequence_pred_new/X_pairplot.png")
#%%
# pairgrid
def my_hist(x, label, color):
    ax0 = plt.gca()
    ax = ax0.twinx()
    
    sns.despine(ax=ax, left=True, top=True, right=False)
    ax.yaxis.tick_right()
    ax.set_ylabel('Counts')
    
    ax.hist(x, label=label, color=color)

sns_plot = sns.PairGrid(data=df_temp.sample(1000), 
    vars=plot_vars,
    hue='Sequence', 
    hue_order=['t1', 't2', 'flair', 'dwi','swi', 'other', 'none_nid'],
    hue_kws={'markers':markers_dic}, palette='Set1', corner=True,
     )
sns_plot.map_diag(my_hist)
sns_plot.map_lower(sns.scatterplot)
# sns_plot._legend.set_title('Class')
labels_dic = {'EchoTime':r'$t_\mathrm{echo}\,[\mathrm{ms}]$', 
    'RepetitionTime':r'$t_\mathrm{rep}\,[\mathrm{ms}]$',
    'InversionTime':'Inversion Time [ms]', 'FlipAngle':r'$\alpha_{\mathrm{flip}} \,[°]$',
    'EchoTrainLength':'ETL', '':'',
    'ImagingFrequency':r'$f_\mathrm{Im}\,[\mathrm{MHz}]$',
    'dBdt':r'$\frac{dB}{dt}\,[\mathrm{T}/\mathrm{s}]$'}

print('Limits have to be adjusted, and axis numbers too')
num_plot_vars = len(plot_vars)
ax_mat = np.tril(np.reshape(np.arange(num_plot_vars**2), (num_plot_vars, num_plot_vars)))
ax_mat = ax_mat[ax_mat!=0]
for i, ax in enumerate(sns_plot.axes.flatten()):
    if i in ax_mat:
        ax.set_xlabel(labels_dic[ax.get_xlabel()], fontsize=20)
        ax.set_ylabel(labels_dic[ax.get_ylabel()], fontsize=20)
        ax.tick_params(labelsize=15)

sns_plot.savefig(f"{fig_dir}/sequence_pred_new/X_pairplot_alt.png")
plt.clf() # Clean parirplot figure from sns 
Image(filename=f"{fig_dir}/sequence_pred_new/X_pairplot_alt.png")




#%%
# In[Lets set missing values to the median]
# df_all[TI_k] = np.where((df_all[TI_k].isna()), df_all[TI_k].median(), df_all[TI_k])
numeric_cols = ['dBdt','EchoTime', 'RepetitionTime','FlipAngle',
    'EchoTrainLength', 'EchoNumbers','ImagingFrequency',]
for num_col in numeric_cols:
    if num_col!=EN_k:
        df_all[num_col] = np.where(
            (df_all[num_col].isna()), df_all[num_col].median(), 
            df_all[num_col])
# Set to one where echo number is missing, discrete, most often occ value
df_all[EN_k] = np.where(
    (df_all[EN_k].isna()), 1, df_all[EN_k])

#%%
# Missing values
print("Missing values:")
print(df_all[numeric_cols].isna().sum(axis=0))
df_all.to_feather(join('data','xgb_sequence_pred', 'preprocessed.feather'))







#####################################################################################################

#%%
# In[Load preprocessed dataframe]
df_all = pd.read_feather(join('data','xgb_sequence_pred', 'preprocessed.feather'))
seq_count = df_all[sq].value_counts()
df_all.keys()
df_all.isna().mean(axis=0)
#%%
# In[Define Columns that will be used for prediction]
print(df_all.keys())
numeric_cols = ['dBdt','EchoTime', 'RepetitionTime','FlipAngle',
    'EchoTrainLength', 'EchoNumbers','ImagingFrequency',]
other_cols = ['SeriesInstanceUID', 'StudyInstanceUID', 'PatientID',
       'SeriesDescription', 'Positive','Sequence',]
binary_cols = set(df_all.keys()).difference(set(other_cols))\
    .difference(set(numeric_cols))
print(binary_cols)
print(numeric_cols)
print(len(binary_cols),'binary cols')
print(len(numeric_cols),'numeric cols')
#%%
# In[Show the binary columns]
df_all[binary_cols].mean(axis=0).sort_values()
len(df_all[binary_cols.union(numeric_cols)].keys())
#%%
print('Later show countplot of binary columns, separated by axis')
bin_counts = df_all[binary_cols].sum(axis=0)/len(df_all)
# svis.bar(bin_counts.keys(), bin_counts.values,
            #   figsize=(15, 6), kwargs={'logscale':True},
            #   figname=f"{fig_dir}/sequence_pred_new/X_distr_bin.png")
plt.rcParams['figure.dpi'] = 200
fig, ax = plt.subplots(1,2,figsize=(10,12), )
bin_counts.sort_values()[31:].plot(kind='barh', ax=ax[0],)
bin_counts.sort_values()[2:31].plot(kind='barh', ax=ax[1],)
ax[0].tick_params(axis='x', labelsize=18)
ax[1].tick_params(axis='x', labelsize=18)
ax[0].tick_params(axis='y', labelsize=15)
ax[1].tick_params(axis='y', labelsize=15)
ax[0].set_xlabel(r'$n/N$', fontsize=22)
ax[1].set_xlabel(r'$n/N$', fontsize=22)
ax[0].set_ylabel('Feature', fontsize=22)
fig.tight_layout()
fig.savefig(f"{fig_dir}/sequence_pred_new/X_distr_bin.png")
#%%
# In[Before encoding we have to split up the sequences we want to predict (test set)]
df_test = df_all[df_all.Sequence == 'none_nid']
df_train = df_all[df_all.Sequence != 'none_nid']
del df_all


#%%
# In[Integer encode labels]
print('Encode Labels as integers')
target_labels = df_train.Sequence.unique()
target_ids = np.arange(len(target_labels))
target_dict = dict(zip(target_labels, target_ids))
y = df_train[sq].map(target_dict)
print(y)
print(target_dict)


#%%
# In[Now we separate the patient ID and the SeriesInstance UID]
# From now on we should not change the order or remove any values,
# otherwise the ids wont match afterwards
# also contains sequences
df_train_ids = df_train[[PID_k, SID_k, sq,]]
df_test_ids = df_test[[PID_k, SID_k,]]
df_train = df_train[list(numeric_cols)+list(binary_cols)]
df_test = df_test[list(numeric_cols)+list(binary_cols)]
print('Actual training columns ',df_train.keys())
#%%
# In[Define X and X_test]
X = df_train.copy()
X_test = df_test.copy()
X_arr = X.to_numpy()
y_arr = y.to_numpy()
#%%
# In[PCA]
y_pca = np.expand_dims(y_arr, axis=0).T
vis_n = len(X_arr)
X_pca = PCA(n_components=2).fit_transform(X_arr)
#%%
# In[Vis PCA]

#plt.rcParams['figure.dpi'] = 250
print(target_dict)
nice_labels_dic = {'flair':'FLAIR','t2':'T2', 'other':'OIS','t1':'T1','swi':'SWI','dwi':'DWI'}
inv_tgt_dic = {v: k for k, v in target_dict.items()}

print('Nice, some clusters can be seen, work on this')
df_pca = pd.DataFrame(data=np.concatenate((X_pca,y_pca), axis=1), 
    columns=['PC 1', 'PC 2', 'Class'])#
fig, ax = plt.subplots()
sns_plot = sns.scatterplot(data=df_pca, x='PC 1', y='PC 2', 
    hue='Class',palette='Set1',s=3, ax=ax,alpha=.4)

h, labels= ax.get_legend_handles_labels()
labels = [nice_labels_dic[inv_tgt_dic[int(float(l))]] for l in labels]
ax.legend(handles = h, labels=labels)
ax.set_xlim(-3500,10500)
ax.set_ylim(-150,430)
# ax.tick_params(axis='both', labelsize=18)

fig.tight_layout()
fig.savefig(f"{fig_dir}/sequence_pred_new/PCA.png")

#%%
# In[tsne embedding]
# Apply PCA first to reduce number of features to 30
print('Reduce number of dimensions to 30')
X_pca30 = PCA(n_components=5).fit_transform(X_arr)
print('Scale data')
X_arr_s = StandardScaler().fit_transform(X_pca30)
n_samples = 1000
rng = default_rng()
inds = rng.choice(len(X_arr), size=n_samples, replace=False)
print('tsne embed')
X_tsne = TSNE(n_components=2, perplexity=20, learning_rate='auto',
                   init='random').fit_transform(X_arr_s[inds])
fig, ax = plt.subplots()
y_tsne = np.expand_dims(y_arr[:n_samples], axis=0).T
df_tsne = pd.DataFrame(data=np.concatenate((X_tsne,y_tsne), axis=1), 
    columns=['Dim 1', 'Dim 2', 'Class'])
sns.scatterplot(data=df_tsne, x='Dim 1', y='Dim 2', hue='Class',palette='Set1')
print('Cannot get tsne to work')






#%%
# In[Split train and val]
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42)
print('X_train shape', X_train.shape)
print('X_val shape', X_val.shape)
#%%
# In[Run bayesian optimization in 5 fold cross validation]
import importlib
importlib.reload(bayesian_opt)
bp = bayesian_opt.find_best_params(X_train.to_numpy(), y_train.to_numpy(),
                                 n_iter=100, nfold=5, n_jobs=5)
print('Best Parameters:')
print(bp)
with open('xgboost/best_params.txt', 'w') as f:
    json.dump(bp, f)

#%%
# In[Initialize and train]
with open('data/xgb_sequence_pred/best_params.txt', 'r') as f:
    bp = json.load(f)

xgb_cl = xgb.XGBClassifier(objective='multi:softprob',
                          tree_method='auto',
                          eval_metric='mlogloss',
                          use_label_encoder=False,
                          **bp)

xgb_cl.fit(X_train, y_train)
xgb_cl.save_model("data/xgb_sequence_pred/trained_classifier.json")




#%%
# Load existing model
with open('data/xgb_sequence_pred/best_params.txt', 'r') as f:
    bp = json.load(f)
xgb_cl = xgb.XGBClassifier(objective='multi:softprob',
                          tree_method='auto',
                          eval_metric='mlogloss',
                          use_label_encoder=False,
                          **bp)
xgb_cl.load_model("data/xgb_sequence_pred/trained_classifier.json")
#%%
# In[Plot feature importance]
# Get the importance dictionary (by gain) from the booster
importance = xgb_cl.get_booster().get_fscore()
# make your changes
for key in importance.keys():
    importance[key] = round(importance[key],4)
fig, ax = plt.subplots(figsize=(8,5))
# provide the importance dictionary to the plotting function
ax = xgb.plot_importance(importance, max_num_features=10, show_values=True,ax=ax)
# xgb.plot_importance(xgb_cl, importance_type = 'gain',max_num_features=10,) # other options available
fig.tight_layout()
fig.savefig(f"{fig_dir}/sequence_pred_new/feature_importance.png")
#%%
# In[ROC curves]
pred_prob_val = xgb_cl.predict_proba(X_val)

fig, ax = plt.subplots()
rc = skplot.metrics.plot_roc_curve(y_val, pred_prob_val, ax=ax,)
ax.legend(labels=['FLAIR', 'T2', 'OIS','T1','SWI','DWI', 'micro-average', 'macro_average'],
    loc='lower right', fontsize=15, ncol=2)
ax.set_xlabel('False Positive Rate', fontsize=18)
ax.set_ylabel('True Positive Rate', fontsize=18)
ax.set_title('')
ax.tick_params(labelsize=14)
fig.tight_layout()
fig.savefig(f"{fig_dir}/sequence_pred_new/ROC_curves.png")


#%%

#%%
# In[Test FPR for different thresholds, Obsolete]
"""
thresholds = np.linspace(.5, .99, 100)
fprs = []
for i, th in enumerate(thresholds):
    pred_val = clss.prob_to_class(pred_prob_val, th, 0)
    cm = confusion_matrix(y_val, pred_val)
    fprs.append(clss.fpr_multi(cm))
fprs = np.array(fprs)


final_th = 0.92
fig, ax = plt.subplots(figsize=(9, 5))
for i in y.unique():
    ax.plot(thresholds, fprs[:, 1+i], label=list(target_dict.keys())[i+1])
ax.axvline(final_th, color='red', linestyle='--',
           label=f'final cutoff={final_th}')
lgd = ax.legend(fontsize=16.5, facecolor='white', labels=['T2','OIS','T1','SWI','DWI','T2*', 
            'Cutoff'])

ax.set_xlabel('Cutoff', fontsize=20)
ax.set_ylabel('FPR', fontsize=20)
fig.tight_layout()
fig.savefig(f"{fig_dir}/sequence_pred/fpr_cutoff.png", dpi=80)
"""
#%%
# In[Plot confusion matrix]
plt.rcParams['figure.dpi'] = 200
fig, ax = plt.subplots()
pred_val = np.argmax(pred_prob_val, axis=1)

ax = skplot.metrics.plot_confusion_matrix(y_val, pred_val, normalize=True, ax=ax)
nice_labels_dic = {'flair':'FLAIR','t2':'T2', 'other':'OIS','t1':'T1','swi':'SWI','dwi':'DWI'}
ax.set_xticklabels([nice_labels_dic[k] for k in target_dict.keys()])
ax.set_yticklabels([nice_labels_dic[k] for k in target_dict.keys()])
ax.tick_params(labelsize=12)
ax.set_xlabel('Predicted Label', fontsize=18)
ax.set_ylabel('True Label', fontsize=18)
ax.set_title('')
fig.tight_layout()
fig.savefig(f"{fig_dir}/sequence_pred_new/confusion_matrix_val_norm.png")

#%%
# In[make prediction for the test set]
nice_labels_dic = {'flair':'FLAIR','t2':'T2', 'other':'OIS','t1':'T1','swi':'SWI','dwi':'DWI'}
pred_prob_test = xgb_cl.predict_proba(X_test)
pred_test = np.argmax(pred_prob_test, axis=1)
# pred_test = clss.prob_to_class(pred_prob_test, final_th, 0)
(labels, counts) = np.unique(pred_test, return_counts=True)

inds = np.argsort(counts)[::-1]
labels = [nice_labels_dic[inv_tgt_dic[k]] for k in labels[inds]]
svis.bar(labels, counts[inds],
         kwargs={'xlabel':'Sequence Type', 'title':'',
                 },
         save=True, figname=f"{fig_dir}/sequence_pred_new/predicted_seq_count.png")


#%%
# In[Get labels and save them]
def dict_mapping(t): return basic.inv_dict(target_dict)[t]
pred_test_labels = np.array([dict_mapping(xi) for xi in pred_test])
print(pred_test_labels)
with open(join('data','xgb_sequence_pred', 'pred_test_labels.pkl'), 'wb') as f:
    pickle.dump(pred_test_labels, f)
#%%
# In[show predicted and true]
print('Sequence count needed for this plot')
plt.style.use('ggplot')                                                          
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']   


labels_pre, counts_pre = seq_count.keys(), seq_count.values
# remove none_nid
mask = labels_pre!='none_nid'
labels_pre, counts_pre = labels_pre[mask], counts_pre[mask] 
labels_post, counts_post = np.unique(pred_test_labels, return_counts=True)

# get both lists in same order
indexes = defaultdict(deque)
for i, x in enumerate(labels_pre):
    indexes[x].append(i)
ids = [indexes[x].popleft() for x in labels_post]
labels_pre, counts_pre =  labels_pre[ids], counts_pre[ids]
labels = np.array([nice_labels_dic[k] for k in labels_pre])
# sort lists starting with largest
sort_inds = [2,4,5, 1, 0, 3]
labels = labels[sort_inds]
counts_post =  counts_post[sort_inds] 
counts_pre =  counts_pre[sort_inds] 

fig, ax = svis.bar(labels, counts_pre, label='True',color=colors[5])
fig, ax = svis.bar(labels, counts_post, fig=fig, ax=ax,
              bottom=counts_pre, label='Predicted', color=colors[3], 
              kwargs={'save_plot':False, 'lgd':True, 'xlabel':'Class',
                      'color':(1,3), 
                      'title':'',
                      'ylabel':'# Scans',
                      },)
fig.savefig(f"{fig_dir}/sequence_pred_new/old_and_predicted_seq_count.png")



#%%
# In[Concat and save]
df_test_pred = df_test_ids
df_test_pred[sq] = pred_test_labels
#df_test_pred
df_pred_seqs = pd.concat([df_test_pred, df_train_ids], axis=0)
df_final = pd.merge(df_init, df_pred_seqs[[SID_k, sq]], on=SID_k, how='inner')

assert len(df_final)==len(df_pred_seqs)
#%%
# In[Examine and save final df]
df_final.to_csv(
    f"{table_dir}/scan_tables/scan_after_sq_pred.csv", index=False)
df_final.to_pickle(
    f"{table_dir}/scan_tables/scan_after_sq_pred.pkl")
print(len(df_final))
#%%


#%%
# In[show predicted and true]
print('Patient count needed for this plot')
                                                        
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']   
pat_count_pre = df_train_ids.groupby(sq).PatientID.nunique()
pat_count_post = df_test_pred.groupby(sq).PatientID.nunique()
labels_pre, counts_pre = pat_count_pre.keys(), pat_count_pre.values
labels_post, counts_post = pat_count_post.keys(), pat_count_post.values
sort_inds = np.argsort(pat_count_pre.values+pat_count_post.values)[::-1]
labels_post, counts_post = labels_post[sort_inds], counts_post[sort_inds] 
labels_pre, counts_pre = labels_pre[sort_inds], counts_pre[sort_inds] 
print('old', labels_pre, 'new',labels_post)
inds = np.arange(1,len(labels_pre))
labels = [nice_labels_dic[k] for k in labels_pre[inds]]
fig, ax = svis.bar(labels, counts_pre[inds], label='Initial',color=colors[5])
fig, ax = svis.bar(labels, counts_post[inds], fig=fig, ax=ax,
              bottom=counts_pre[inds], label='After Prediction', color=colors[3], 
              kwargs={'save_plot':False, 'lgd':True, 'xlabel':'Class',
                      'color':(1,3),  
                      'title':'',
                      'ylabel':'# Patients',
                      },)
fig.savefig(f"{fig_dir}/sequence_pred_new/old_and_predicted_pat_count.png")
#%%

#%%
# In[Same but for positive patients]
pos_pat = df_init[df_init.Positive==1].PatientID.unique()
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']   
df_train_ids_p = df_train_ids[df_train_ids.PatientID.isin(pos_pat)]
df_test_pred_p = df_test_pred[df_test_pred.PatientID.isin(pos_pat)]
pat_count_pre = df_train_ids_p.groupby(sq).PatientID.nunique()
pat_count_post = df_test_pred_p.groupby(sq).PatientID.nunique()
labels_pre, counts_pre = pat_count_pre.keys(), pat_count_pre.values
labels_post, counts_post = pat_count_post.keys(), pat_count_post.values
sort_inds = np.argsort(pat_count_pre.values+pat_count_post.values)[::-1]
labels_post, counts_post = labels_post[sort_inds], counts_post[sort_inds] 
labels_pre, counts_pre = labels_pre[sort_inds], counts_pre[sort_inds] 
print('old', labels_pre, 'new',labels_post)
inds = np.arange(1,len(labels_pre))
labels = [nice_labels_dic[k] for k in labels_pre[inds]]
fig, ax = svis.bar(labels, counts_pre[inds], label='Initial',color=colors[5])
fig, ax = svis.bar(labels, counts_post[inds], fig=fig, ax=ax,
              bottom=counts_pre[inds], label='After Prediction', color=colors[3], 
              kwargs={'save_plot':False, 'lgd':True, 'xlabel':'Class',
                      'color':(1,3),'yrange':(0,440),  
                      'title':'',
                      'ylabel':'# Positive Patients',
                      },)
fig.savefig(f"{fig_dir}/sequence_pred_new/old_and_predicted_pospat_count.png")

fig2,ax2 = plt.subplots(figsize=(8,4))
handles, labels = ax.get_legend_handles_labels()
fig2.legend(handles, labels, loc='center', fontsize=40)
plt.grid(b=None)
plt.axis('off')
plt.show()
fig2.tight_layout()
fig2.savefig(f"{fig_dir}/sequence_pred_new/legend.png")
#%%
df_test_pred.groupby(sq).count()
#%%
mpl.rcParams['figure.dpi'] = 400
labels = ['other', 'T1', 'T2', 'FLAIR', 'DWI', 'SWI', 'T2*']
g = sns.countplot(x="Sequence", hue="Positive", data=df_final, hue_order=[1,0],
    order = df_final['Sequence'].value_counts().index)
g.set(xlabel=('Sequence Type'), ylabel=('Volume Count'), )
fig = g.get_figure()
fig.tight_layout()
fig.savefig(f"{fig_dir}/sequence_pred/volumes_sequence_count_predicted.png")
#%%
# In[Show predicted and true grouped by patients]
pos_mask = df_final.Positive==1
pos_pat_count_seq = df_final[pos_mask].groupby(by='Sequence')\
    .PatientID.nunique().sort_values(ascending=False)
neg_pat_count_seq = df_final[~pos_mask].groupby(by='Sequence')\
    .PatientID.nunique().sort_values(ascending=False)
n_labels = ['Other', 'Flair', 'T2',  'DWI', 'T1', 'SWI', 'T2*']
fig, ax = svis.bar(n_labels, neg_pat_count_seq.values, 
                   label='neg', color=svis.Color_palette(0)[1])

svis.bar(n_labels,pos_pat_count_seq.values, fig=fig, ax=ax,
         bottom=neg_pat_count_seq.values, label='pos',
         color=svis.Color_palette(0)[0], 
         kwargs={'lgd':True, 'xlabel':'Sequence Type','title':'All Patients',
                 'ylabel':'Patient Count'},
         save=True, 
         figname=f"{fig_dir}/sequence_pred/seq_patient_count_predicted.png")


# In[https://towardsdatascience.com/beginners-guide-to-xgboost-for-classification-problems-50f75aac5390]