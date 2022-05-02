# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 15:50:50 2021

@author: klein
"""
#%%
from collections import defaultdict, deque
import os
import pickle
import json
from os.path import join
from pathlib import Path
import pandas as pd
from sympy import count_ops
from utilities import basic, mri_stats
from stats_tools import vis as svis
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
import scikitplot as skplot
from numpy.random import default_rng
import proplot as pplt
#import proplot as pplt
# params = {'figure.dpi':300,
        # 'legend.fontsize': 18,#
        # 'figure.figsize': [8, 5],
        #  'axes.labelsize': 18,
        #  'axes.titlesize':18,
        #  'xtick.labelsize':18,
        #  'ytick.labelsize':18}
#pylab.rcParams.update(params)
pplt.rc.cycle = 'ggplot'
colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] 
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
df_final = pd.read_pickle(f"{table_dir}/scan_tables/scan_after_sq_pred_dst.pkl")
df_final = df_final.drop_duplicates()
df_final = df_final[['PatientID','SeriesInstanceUID','Sequence','days_since_test']]
# df_init = pd.read_csv(
    # join(table_dir, 'neg_pos_clean.csv'), nrows=80000)
#df_init.Positive = 0 
#df_init.loc[df_init.days_since_test>=-4, 'Positive'] = 1
#%%
# In[Load preprocessed dataframe]
df_all = pd.read_feather(join('data','xgb_sequence_pred', 'preprocessed.feather'))
df_all = df_all[['PatientID','SeriesInstanceUID','Sequence']]
df_all = pd.merge(df_all, df_final[['SeriesInstanceUID','days_since_test']], on='SeriesInstanceUID', how='left')
nice_labels_dic = {'flair':'FLAIR','t2':'T2', 'other':'Other','t1':'T1','swi':'SWI','dwi':'DWI'}
seq_count = df_all[sq].value_counts()
#%%


# All scans
seq_count = df_all[sq].value_counts()
target_labels = df_all.Sequence.unique()
target_ids = np.arange(len(target_labels))
target_dict = dict(zip(target_labels, target_ids))
def dict_mapping(t): return basic.inv_dict(target_dict)[t]
labels_pre, counts_pre = seq_count.keys(), seq_count.values
# remove none_nid
with open(join('data','xgb_sequence_pred', 'pred_test_labels.pkl'), 'rb') as f:
    pred_test_labels = pickle.load(f)
# counts all scans
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
sort_inds = [4, 5, 1, 3, 0] #include 2 if you want to show 'other scans'
labels = labels[sort_inds]

counts_post =  counts_post[sort_inds] 
counts_pre =  counts_pre[sort_inds] 



fig, axs = pplt.subplots(ncols=2, nrows=2, figwidth=5,  xlabel='Sequence Type',
    abc=True,sharex=2, sharey=0)

hs = []
h = axs[0].bar(labels, counts_pre, label='true',color=colors[5])
hs.append(h)
h = axs[0].bar(labels, counts_post, bottom=counts_pre, label='predicted', color=colors[3])
hs.append(h)
axs[0].set_ylabel('# Scans')
axs[2].set_ylabel('# Patients')

fig.legend(hs, ncols=2, center=True, frame=False, loc='b', col=2)



rel_seq = ['t1','t2','flair','swi','dwi',]
pat_count_pre = df_all.groupby(sq).PatientID.nunique()[rel_seq]
#print(pat_count_pre)
pat_count_post = df_final.groupby(sq).PatientID.nunique()[rel_seq]
#print(pat_count_post)
labels_pre, counts_pre = pat_count_pre.keys(), pat_count_pre.values
labels_post, counts_post = pat_count_post.keys(), pat_count_post.values
labels = [nice_labels_dic[k] for k in labels_pre]
#print(labels)
#print(counts_pre)
axs[2].bar(labels, counts_pre, label='true',color=colors[5])
axs[2].bar(labels, counts_post-counts_pre, bottom=counts_pre, 
   label='predicted', color=colors[3])
#width = .8

df_all_p = df_all[df_all.days_since_test>=-4]
df_fin_p = df_final[df_final.days_since_test>=-4]
seq_counts_pre = df_all_p.groupby(sq).SeriesInstanceUID.nunique()[rel_seq]
#print(pat_count_pre)
seq_counts_post = df_fin_p.groupby(sq).SeriesInstanceUID.nunique()[rel_seq]

axs[1].bar(labels, seq_counts_pre, color=colors[5])
axs[1].bar(labels, seq_counts_post-seq_counts_pre, bottom=seq_counts_pre, 
    color=colors[3])
axs[1].set_ylabel('# Positive Scans')



counts_pre = df_all_p.groupby(sq).PatientID.nunique()[rel_seq]
#print(pat_count_pre)
counts_post = df_fin_p.groupby(sq).PatientID.nunique()[rel_seq]

axs[3].bar(np.arange(len(labels))-.2, counts_pre, color=colors[5], align='center', width=.4)
axs[3].bar(np.arange(len(labels))-.2, counts_post-counts_pre, bottom=counts_pre, 
    color=colors[3],width=.4, align='center')
counts_pre_post = {}

for seq in rel_seq:
    df_seq = df_final[df_final[sq]==seq]
    pos_pat = df_seq[df_seq.days_since_test>=-4].PatientID.unique()
    df_seq_pre_post = df_seq[
        (df_seq.days_since_test<=-30) & (df_seq.PatientID.isin(pos_pat))]
    counts_pre_post[seq] = df_seq_pre_post.PatientID.nunique()
counts_pre_post['t1'] = 35

h = axs[3].bar(np.arange(len(counts_pre_post))+.2, 
    counts_post-list(counts_pre_post.values()), 
    color=colors[6], 
    align='center', width=.4, label='pos.')
hs.append(h)
h = axs[3].bar(np.arange(len(labels))+.2,  list(counts_pre_post.values()),
    bottom=counts_post-list(counts_pre_post.values()), 
    color=colors[0],width=.4, align='center', label='pos. and neg.')
hs.append(h)
axs[3].set_ylabel('# Positive Patients')

fig.legend(hs, ncols=2, center=True, frame=False, loc='b', col=2)
#axs[3].bar(np.arange(len(labels))-.2, counts_post-counts_pre, bottom=counts_pre, 
 #   color=colors[3],width=.4, align='center')
#axs[3].bar(np.arange(len(labels))+.2, counts_pre, color=colors[5], align='center', width=.4)
#axs[3].bar(np.arange(len(labels))+.2, counts_post-counts_pre, bottom=counts_pre, 
#    color=colors[3],width=.4, align='center')

#axs[3].set_ylabel('# Positive Patients')




#fig.savefig(f"{fig_dir}/sequence_pred_new/scan_and_pat_count.png", dpi=1000)


#%%
# In[Count number of volumes for every sequence]
seq_count = df_all[sq].value_counts()
print(seq_count)
# In[Get labels and save them]
target_labels = df_all.Sequence.unique()
target_ids = np.arange(len(target_labels))
target_dict = dict(zip(target_labels, target_ids))
def dict_mapping(t): return basic.inv_dict(target_dict)[t]
with open(join('data','xgb_sequence_pred', 'pred_test_labels.pkl'), 'rb') as f:
    pred_test_labels = pickle.load(f)
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
sort_inds = [2,4,5, 1, 3, 0]
labels = labels[sort_inds]

counts_post =  counts_post[sort_inds] 
counts_pre =  counts_pre[sort_inds] 
fig, ax = plt.subplots(figsize=(3,3))
fig, ax = svis.bar(labels, counts_pre, label='',color=colors[5])
fig, ax = svis.bar(labels, counts_post, fig=fig, ax=ax,
              bottom=counts_pre, label='', color=colors[3], 
              kwargs={'save_plot':False, 'lgd':True, 'xlabel':'Class',
                      'color':(1,3), 
                      'title':'',
                      'ylabel':'# Scans',
                      },)
fig.savefig(f"{fig_dir}/sequence_pred_new/old_and_predicted_seq_count.png")

#%%
fig, ax = plt.subplots()
fig, ax = svis.bar(np.arange(2), [10,20], ax=ax, fig=fig)
svis.bar(np.arange(2), [2,5], ax=ax, fig=fig, color='k', bottom=[10,20])

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
#df_final.to_csv(
#    f"{table_dir}/scan_tables/scan_after_sq_pred.csv", index=False)
#df_final.to_pickle(
#    f"{table_dir}/scan_tables/scan_after_sq_pred.pkl")
print(len(df_final))
#%%

#%%
# sns.countplot(df_final.Sequence)
#print(df_final[df_final.Sequence=='swi'].PatientID.nunique())
df_test_pred[df_test_pred.Sequence=='swi'].PatientID.nunique()
#%%
# In[show predicted and true]
print('Patient count needed for this plot')
                                                        
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']   
pat_count_pre = df_train_ids.groupby(sq).PatientID.nunique()
pat_count_post = df_final.groupby(sq).PatientID.nunique()
labels_pre, counts_pre = pat_count_pre.keys(), pat_count_pre.values
labels_post, counts_post = pat_count_post.keys(), pat_count_post.values
sort_inds = sort_inds = [2,4,5,1,3,0]#np.argsort(pat_count_pre.values+pat_count_post.values)[::-1]
labels_post, counts_post = labels_post[sort_inds], counts_post[sort_inds] 
labels_pre, counts_pre = labels_pre[sort_inds], counts_pre[sort_inds] 
print('old', labels_pre, 'new',labels_post)
inds = np.arange(1,len(labels_pre))
labels = [nice_labels_dic[k] for k in labels_pre[inds]]
fig, ax = svis.bar(labels, counts_pre[inds], label='',color=colors[5])
fig, ax = svis.bar(labels, counts_post[inds]-counts_pre[inds], fig=fig, ax=ax,
              bottom=counts_pre[inds], label='', color=colors[3], 
              kwargs={'save_plot':False, 'lgd':True, 'xlabel':'Class',
                      'color':(1,3),  
                      'title':'',
                      'ylabel':'# Patients',
                      },)
    
fig.savefig(f"{fig_dir}/sequence_pred_new/old_and_predicted_pat_count.png")
#%%
df_final[df_final.Sequence=='t1'].PatientID.nunique()
#%%

#%%
# In[Same but for positive patients]
pos_pat = df_init[df_init.Positive==1].PatientID.unique()
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']   
df_train_ids_p = df_train_ids[df_train_ids.PatientID.isin(pos_pat)]
df_test_pred_p = df_final[df_final.PatientID.isin(pos_pat)]
pat_count_pre = df_train_ids_p.groupby(sq).PatientID.nunique()
pat_count_post = df_test_pred_p.groupby(sq).PatientID.nunique()
labels_pre, counts_pre = pat_count_pre.keys(), pat_count_pre.values
labels_post, counts_post = pat_count_post.keys(), pat_count_post.values
sort_inds = [2,4,5,1,3,0]
labels_post, counts_post = labels_post[sort_inds], counts_post[sort_inds] 
labels_pre, counts_pre = labels_pre[sort_inds], counts_pre[sort_inds] 
print('old', labels_pre, 'new',labels_post)
inds = np.arange(1,len(labels_pre))
labels = [nice_labels_dic[k] for k in labels_pre[inds]]
save_legend = False
if save_legend:
    label0 = 'Initial'
    label1 = 'After Prediction'
else:
    label0=label1 =''
fig, ax = svis.bar(labels, counts_pre[inds], label=label0,color=colors[5])
fig, ax = svis.bar(labels, counts_post[inds]-counts_pre[inds], fig=fig, ax=ax,
              bottom=counts_pre[inds], label=label1, color=colors[3], 
              kwargs={'save_plot':False, 'lgd':True, 'xlabel':'Class',
                      'color':(1,3),'yrange':(0,330),  
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
if save_legend:
    fig2.savefig(f"{fig_dir}/sequence_pred_new/legend.png")

#%%
df_init.days_since_test
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