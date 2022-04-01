# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 15:50:50 2021

@author: klein
"""
#%%
import xgboost as xgb
import os
from pathlib import Path
import pandas as pd
from utilities import basic, mri_stats
from utilities import classification as clss
from stats_tools import vis as svis
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import scikitplot as skplot
from sklearn.manifold import TSNE
from numpy.random import default_rng
from sklearn.decomposition import PCA
#%%
# In[tables directories]
script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parent
fig_dir = f"{base_dir}/figs/basic_stats"
table_dir = f"{base_dir}/data/tables"
fig_dir = f"{base_dir}/figs"

#%%
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
DT_k = 'DateTime'
ICD_k = 'InstanceCreationDate'
SN_k = 'SequenceName'
PSN_k = 'PulseSequenceName'
SO_l = 'ScanOptions' # List of values like FS, PFP ,...
#%%
# In[load all csv]
df_init = pd.read_pickle(f"{table_dir}/scan_final.pkl")
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
rel_cat_feat_ls = ['ImageType','ScanningSequence', 'SequenceVariant', 'ScanOptions']
other_rel_cols_ls = ['SeriesInstanceUID', 'StudyInstanceUID','PatientID',
                        'Positive']
df_all = df_all[other_rel_cols_ls + rel_numeric_feat_ls + rel_cat_feat_ls]

#%% 
#In [Pulse Sequence Name, unique str, as next step create one
#   hot encoded features for every one of these columns]
psn_strings = ['tse', 'fl',  'tir', 'swi', 'ep', 'se', 'tfl', 're', 'spc', 'pc',
               'SE','FSE','FIR','GE', 'h2', 'me', 'qD', 'ci', 'fm','de','B1']
mask = df_all.SequenceName.str.startswith('*'+psn_strings[0])
mask = mask | df_all.SequenceName.str.startswith(psn_strings[0])
mask = mask | df_all.SequenceName.str.startswith('*q'+psn_strings[0])
for psn_string in psn_strings[1:]:
    mask = mask | df_all.SequenceName.str.startswith('*'+psn_string)
    mask = mask | df_all.SequenceName.str.startswith(psn_string)
    mask = mask | df_all.SequenceName.str.startswith('*q'+psn_string)
print(df_all[~mask].SequenceName.unique())
#%%
# In[Select only relevant columns]
print(f"all elements {len(df_all)}")
df_all = df_all[rel_cols].dropna(subset=[SID_k])
df_all[SS_k][df_all[SS_k].isna()] = ['No_SS',]
df_all[SV_k][df_all[SV_k].isna()] = ['No_SV',]
print(f"after dropping nans in Sequence id: {len(df_all)}")
print("Number of missing values for every column")
print(df_all.isna().sum(axis=0))
print(df_all[SS_k].astype(str).unique())
print(df_all[SV_k].astype(str).unique())
# In[Get masks for the different series descriptions]
mask_dict, tag_dict = mri_stats.get_masks_dict(df_all)

#%% 
print(df_all.PatientID.nunique())
#%%
# In[Add sequence column and set it to one of the relevant values]
sq = 'Sequence'
rel_keys = ['t1', 't1gd', 't2', 't2gd', 't2s', 'swi', 'flair', 'none_nid',
            'gd', 'dwi']
rel_masks = [mask_dict[key] for key in rel_keys]
df_all[sq] = "other"
for mask, key in zip(rel_masks, rel_keys):
    df_all[sq] = np.where((mask), key, df_all[sq])
df_all[sq] = np.where((mask_dict.t1gd), "t1", df_all[sq])
df_all[sq] = np.where((mask_dict.t2gd), "t2", df_all[sq])
df_all[sq] = np.where((mask_dict.gd), "none_nid", df_all[sq])
print(df_all[sq])
print(mask_dict.gre.sum())

#%% 
# print some numbers of scans
rel_seq = [ 't2', 't2s', 'swi', 'flair', 'dwi']
for name, mask in mask_dict.items():    
    if name in rel_seq:
        print(name, ':',mask.sum())
print('other',(mask_dict.more|mask_dict.mip| mask_dict.bold| mask_dict.loc|mask_dict.b1calib|
    mask_dict.autosave|mask_dict.screensave|mask_dict.pd| mask_dict.angio|mask_dict.survey|
    mask_dict.cest|mask_dict.asl|mask_dict.tracew|mask_dict.stir|mask_dict.adc|
    mask_dict.pwi|mask_dict.dti).sum())
print('t1:',(mask_dict.t1gd|mask_dict.t1).sum())
#print("There are not enough dti sequences")
#%%
# In[Count number of volumes for every sequence]
seq_count = df_all[sq].value_counts()
print(seq_count)
print(np.sum(seq_count.values), len(df_all),'no intersections left')
#%%
# In[Plot sequence counts]
mpl.rcParams['figure.dpi'] = 400
plt.style.use('ggplot')
labels = ['Not \nidentified', 'other', 'T1', 'T2', 'DWI', 'FLAIR', 'SWI', 'T2*']
g = sns.countplot(x="Sequence", hue="Positive", data=df_all, hue_order=[1,0],
    order = df_all['Sequence'].value_counts().index)
g.set(xlabel=('Sequence Type'), ylabel=('Volume Count'), xticklabels=labels)
fig = g.get_figure()
fig.tight_layout()
fig.savefig(f"{fig_dir}/sequence_pred/volumes_sequence_count.png")
#%%
# In[PLot class counts, no pos. neg.]
#fig, ax = plt.subplots()

mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['xtick.labelsize']=12
mpl.rcParams['ytick.labelsize']=12
mpl.rcParams.update({'font.size': 1})
print(plt.rcParams['axes.prop_cycle'].by_key()['color'])
plt.style.use('ggplot')
fig, ax = plt.subplots()
labels = ['Unlabeled', 'OIS', 'T1', 'T2', 'DWI', 'FLAIR', 'SWI', 'T2*']
g2 = sns.countplot(x="Sequence", data=df_all, 
    order = df_all['Sequence'].value_counts().index, color='#8EBA42',ax=ax)
# g2.set(xlabel=, ylabel=('# Scans'), xticklabels=labels,)
ax.set_xlabel('Class', fontsize=15)
ax.set_ylabel('# Scans', fontsize=15)
ax.set_xticklabels(labels)
#fig2 = g2.get_figure()
fig.tight_layout()
fig.savefig(f"{fig_dir}/sequence_pred/volumes_sequence_count.png")
print(df_all.Sequence.value_counts())
#%%
# In[plot sequence counts grouped by patients]

pos_mask = df_all.Positive==1
pos_pat_count_seq = df_all[pos_mask].groupby(by='Sequence')\
    .PatientID.nunique().sort_values(ascending=False)
neg_pat_count_seq = df_all[~pos_mask].groupby(by='Sequence')\
    .PatientID.nunique().sort_values(ascending=False)
n_labels = ['Other', 'Flair', 'T2', 'Not\n identified', 'DWI', 'T1', 'SWI', 'T2*']
fig, ax = svis.bar(n_labels, neg_pat_count_seq.values, 
                   label='neg', color=svis.Color_palette(0)[1])

svis.bar(n_labels,pos_pat_count_seq.values, fig=fig, ax=ax,
         bottom=neg_pat_count_seq.values, label='pos',
         color=svis.Color_palette(0)[0], 
         kwargs={'lgd':True, 'xlabel':'Sequence Type','title':'All Patients',
                 'ylabel':'Patient Count'},
         save=True, 
         figname=f"{fig_dir}/sequence_pred/seq_patient_count.png")

#%%
# In[Turn ScanningSequence into multi-hot encoded]
s = df_all[SS_k].explode()
columns_list = list(df_all.columns)
columns_list.remove(SS_k)
df_all = df_all[columns_list].join(pd.crosstab(s.index, s))
print("Unique Scanning Sequences:")
print(set(df_all.columns)-set(columns_list))
del s
#%%
# In[Turn SequenceVariant into multi-hot encoded]
s = df_all[SV_k].explode()
columns_list = list(df_all.columns)
columns_list.remove(SV_k)
df_all = df_all[columns_list].join(pd.crosstab(s.index, s))
print("Unique Sequence Variants:")
print(set(df_all.columns)-set(columns_list))
del s, columns_list
#%% 
# In[plot distribution of the values that we want to predict]
columns_list = list(df_all.columns)
sparse_columns = ['EP', 'GR', 'IR', 'RM', 'SE', 'DE', 'MP', 'MTC',
                  'OSP', 'SK', 'SP', 'SS', 'TOF', 'No_SV', 'No_SS']
rmv_list = [SID_k, PID_k] + sparse_columns #+ ['DateTime', 'Positive','InstanceCreationDate','SeriesDescription'] 

for itm in rmv_list:
    columns_list.remove(itm)
fig, ax = plt.subplots(figsize=(10, 10))
ax = df_all.hist(column=columns_list, figsize=(10, 10), ax=ax, log=False)
fig.tight_layout()
fig.savefig(f"{fig_dir}/sequence_pred/X_distr.png")

#%%
# In[Produce Pairplot]
mpl.rcParams['figure.dpi'] = 300
from IPython.display import Image
markers_dic = dict(zip(['t1', 't2','t2s','flair', 'dwi','swi', 'other', 'none_nid'], 
    ["d", "d", "d", 'd','d','d', 'd', '.']))
sns_plot = sns.pairplot(df_all, 
    vars=['EchoTime', 'RepetitionTime','FlipAngle','InversionTime','EchoTrainLength'],
      diag_kind="hist", hue='Sequence', 
     hue_order=['t1', 't2','t2s','flair', 'dwi','swi', 'other', 'none_nid'],
    plot_kws={"s": 13, 'alpha': 1},
     markers=markers_dic, palette='Set1')
sns_plot._legend.set_title('Class')

labels_dic = {'EchoTime':'Echo Time [ms]', 'RepetitionTime':'Repetition Time [ms]',
        'InversionTime':'Inversion Time [ms]', 'FlipAngle':'Flip Angle [°]',
        'EchoTrainLength':'Echo Train Length', '':''}
for i, ax in enumerate(sns_plot.axes.flatten()):
    ax.set_xlabel(labels_dic[ax.get_xlabel()], fontsize=17)
    ax.set_ylabel(labels_dic[ax.get_ylabel()], fontsize=17)
    if i<5:
        ax.set_ylim(-10,500)
    if i%5==0:
        ax.set_xlim(-10,400)
    if i>19:
        ax
        .set_ylim(-10,300)
    if (i-1)%5==0:
        ax.set_xlim(-400,12500)
    if (i+1)%5==0:
        ax.set_xlim(-10,250)
handles = sns_plot._legend_data.values()
labels = sns_plot._legend_data.keys()
new_labels = ['T1', 'T2', 'T2*','FLAIR', 'DWI','SWI', 'OIS','Unknown']
lgd_labels_dic = dict(zip(['t1','t2','t2s','flair','dwi','swi','other','none_nid'],
    new_labels))
sns_plot._legend.remove()
plt.legend(handles=handles, labels=[lgd_labels_dic[l] for l in labels], 
    loc=(-4.8,5.34), ncol=8, fontsize=16)
sns_plot.savefig(f"{fig_dir}/sequence_pred/X_pairplot.png")
plt.clf() # Clean parirplot figure from sns 
Image(filename=f"{fig_dir}/sequence_pred/X_pairplot.png")

#%%
# In[Fraction of missing values in each column]
print("Number of missing values")
print(df_all.isna().sum(axis=0))
#%%
# In[Lets set missing inversion times to 0]
df_all[TI_k] = np.where((df_all[TI_k].isna()), 0, df_all[TI_k])
df_all[FA_k] = np.where((df_all[FA_k].isna()), 0, df_all[FA_k])
df_all[TE_k] = np.where((df_all[TE_k].isna()), 0, df_all[TE_k])
#%%
# Lets set missing TR to the median TR
df_all[TR_k] = np.where((df_all[TR_k].isna()), df_all[TR_k].median(), df_all[TR_k])
df_all[ETL_k] = np.where((df_all[ETL_k].isna()), 0, df_all[ETL_k])
print("Missing values:")
print(df_all.isna().sum(axis=0))

#%%
# In[Show the binary columns]
bin_counts = df_all[sparse_columns].sum(axis=0)
svis.bar(bin_counts.keys(), bin_counts.values,
              figsize=(15, 6), kwargs={'logscale':True},
              figname=f"{fig_dir}/sequence_pred/X_distr_bin.png")
#%%
# In[We can drop the TOF, EP, DE and MTC columns]
try:
    df_all = df_all.drop(['TOF', 'DE', 'MTC'], axis=1)
except:
    print("those columns are not present")
#%%
# In[Before encoding we have to split up the sequences we want to predict (test set)]
df_test = df_all[df_all.Sequence == 'none_nid']
df_train = df_all[df_all.Sequence != 'none_nid']
del df_all


#%%
# In[Integer encode labels]
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
df_train_ids = df_train[[PID_k, SID_k, sq, DT_k, ICD_k]]
df_test_ids = df_test[[PID_k, SID_k, DT_k, ICD_k]]
try:
    df_train = df_train.drop(
        [PID_k, SID_k, 'Sequence', SD_k, DT_k, ICD_k], axis=1)
    df_test = df_test.drop(
        [PID_k, SID_k, 'Sequence', SD_k, DT_k, ICD_k], axis=1)
except:
    print("those columns are not present")

print('Actual training columns ',df_train.keys())
#%%
# In[Min max scale non bin columns]
num_cols = df_train.select_dtypes(include="float64").columns
X = df_train.copy()
X_test = df_test.copy()
for col in num_cols:
    X[col] = (df_train[col] - df_train[col].min()) / \
        (df_train[col].max()-df_train[col].min())
    X_test[col] = (df_test[col] - df_test[col].min()) / \
        (df_test[col].max()-df_test[col].min())
#%%
# In[PCA]
print('tsne and pca both fail to visualize cluster')
X_arr = X.to_numpy()
y_arr = y.to_numpy()
y_pca = np.expand_dims(y_arr, axis=0).T
X_pca = PCA(n_components=2).fit_transform(X_arr[:2000])
#%%
# In[Vis PCA]
df_pca = pd.DataFrame(data=np.concatenate((X_pca,y_pca[:2000]), axis=1), 
    columns=['PC 1', 'PC 2', 'Class'])
sns_plot = sns.scatterplot(data=df_pca, x='PC 1', y='PC 2', hue='Class',palette='Set1',s=5)

#handles = sns_plot._legend_data.values()
#labels = sns_plot._legend_data.keys()
#sns_plot._legend.remove()
#plt.legend(handles=handles, labels=labels, 
#    loc=(-4.8,5.34), ncol=7, fontsize=16)
#%%
# In[tsne embedding]

n_samples = 100
rng = default_rng()
inds = rng.choice(len(X), size=n_samples, replace=False)
X_tsne = TSNE(n_components=2,perplexity=100, learning_rate='auto',
                   init='random').fit_transform(X_arr[inds])
#%%
# In[tsne visualization]
fig, ax = plt.subplots()
y_tsne = np.expand_dims(y_arr[:n_samples], axis=0).T
df_tsne = pd.DataFrame(data=np.concatenate((X_tsne,y_tsne), axis=1), 
    columns=['Dim 1', 'Dim 2', 'Class'])
sns.scatterplot(data=df_tsne, x='Dim 1', y='Dim 2', hue='Class',palette='Set1')
#%%
# In[Split train and val]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42)
print('X_train shape', X_train.shape)
print('X_val shape', X_val.shape)
#%%
# In[Initialize and train]
xgb_cl = xgb.XGBClassifier()
xgb_cl.fit(X_train, y_train)
#%%
# In[Plot feature importance]
xgb.plot_importance(xgb_cl, importance_type = 'gain') # other options available
plt.show()
#%%
# In[Predict]
pred_prob_val = xgb_cl.predict_proba(X_val)
#%%
# In[Plot roc curve]
svis.plot_decorator(skplot.metrics.plot_roc_curve, 
                    plot_func_args=[y_val, pred_prob_val, ],
                    plot_func_kwargs={'figsize': (9, 8), 'text_fontsize': 14.5,
                            'title': "Sequence Prediction - ROC Curves"},
                    figname=f"{fig_dir}/sequence_pred/ROC_curves.png")
#%%
# In[Test FPR for different thresholds]
thresholds = np.linspace(.5, .99, 100)
fprs = []
for i, th in enumerate(thresholds):
    pred_val = clss.prob_to_class(pred_prob_val, th, 0)
    cm = confusion_matrix(y_val, pred_val)
    fprs.append(clss.fpr_multi(cm))
fprs = np.array(fprs)
#%%
# In[Plot FPR for different thresholds]
# final_th = 0.92
final_th = 0.92
fig, ax = plt.subplots(figsize=(9, 5))
for i in range(6):
    ax.plot(thresholds, fprs[:, 1+i], label=list(target_dict.keys())[i+1])
ax.axvline(final_th, color='red', linestyle='--',
           label=f'final cutoff={final_th}')
lgd = ax.legend(fontsize=16.5, facecolor='white', labels=['T2','OIS','T1','SWI','DWI','T2*', 
            'Cutoff'])

ax.set_xlabel('Cutoff', fontsize=20)
ax.set_ylabel('FPR', fontsize=20)
fig.tight_layout()
fig.savefig(f"{fig_dir}/sequence_pred/fpr_cutoff.png", dpi=80)
#%%
# In[Plot confusion matrix]
# np.argmax(pred_val, axis=1)
#
pred_val = clss.prob_to_class(pred_prob_val, final_th, 0)
cm = confusion_matrix(y_val, pred_val)

plot_func_args = [y_val, pred_val]
plot_func_kwargs = {'normalize': False, 'text_fontsize': 16, 
                    'title_fontsize': 18, }
svis.plot_decorator(skplot.metrics.plot_confusion_matrix, 
                    plot_func_args, plot_func_kwargs,
                    kwargs={'xticks':np.arange(7), 
                            'xtick_labels':target_dict.keys(), 
                            'yticks':np.arange(7), 
                            'ytick_labels':target_dict.keys()},
                    save=True, 
                    figname=f"{fig_dir}/sequence_pred/confusion_matrix_val_norm.png")
#%%
# In[make prediction for the test set]
pred_prob_test = xgb_cl.predict_proba(X_test)
pred_test = clss.prob_to_class(pred_prob_test, final_th, 0)
svis.bar(target_dict.keys(), np.unique(pred_test, return_counts=True)[1],
         kwargs={'xlabel':'Sequence', 'title':'Predicted sequences',
                 },
         save=True, figname=f"{fig_dir}/sequence_pred/seq_dist_pred.png")
#%%
# In[show predicted and true]
pred_counts = np.unique(pred_test, return_counts=True)[1]
fig, ax = svis.bar(target_dict.keys(), seq_count.values[1:], label='true',
                   )
svis.bar(target_dict.keys(), pred_counts, fig=fig, ax=ax,
              bottom=seq_count.values[1:], label='pred', color=(0,1), 
              kwargs={'save_plot':True, 'lgd':True, 'xlabel':'Sequence Type',
                      'color':(1,3), 'yrange':(0,126000), 
                      'title':'Volume Count - All Patients',
                      'ylabel':'Volumes Count'},
              figname=f"{fig_dir}/sequence_pred/seq_dist_pred.png")

#%%
# In[Get labels from prediction]
def dict_mapping(t): return basic.inv_dict(target_dict)[t]

pred_test_labels = np.array([dict_mapping(xi) for xi in pred_test])
#%%
# In[Create and save dataframe with predictions]
df_test[sq] = pred_test_labels
df_test = pd.concat([df_test, df_test_ids], axis=1)
df_train = pd.concat([df_train, df_train_ids], axis=1)
df_test['TrueSequenceType'] = 0
df_train['TrueSequenceType'] = 1
df_final = pd.concat([df_train, df_test])
#%%
#del df_test, df_train
# In[Examine final df]
df_final[[PID_k, SID_k, sq, ICD_k, 'TrueSequenceType']].to_csv(
    f"{base_dir}/share/pred_seq.csv", index=False)
print(len(df_final))
print(df_final.isna().sum())

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