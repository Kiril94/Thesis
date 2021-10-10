# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 10:15:59 2021

@author: klein
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cryptography.fernet import Fernet
from utilities import basic
from stats_tools import vis as svis


# In[Load and decrypt dataframe]
base_dir = "D:/Thesis/Cobra"
fig_dir = f"{base_dir}/figs/cloud_res"
csv_folder = "D:/Thesis/Cobra/share"
df = pd.read_csv(f"{csv_folder}/import/dst.csv").iloc[:, 1:]
df_seq = pd.read_csv(f"{csv_folder}/export/pred_seq.csv").iloc[:, 1:]
key = b'1b3KCzziTwLPiqneoY8XMEQ2DhWpxixIeiRhLIWwZe4='
fernet = Fernet(key)
def decrypt(x): return (fernet.decrypt(
    bytes(x[2:-1], 'utf-8'))).decode('utf-8')


df.DST = df.DST.map(decrypt)
df.DST = df.DST.map(int)
# In[Get patients who had scans pre and post pos. test]
SID = 'SeriesInstanceUID'
Seq = 'Sequence'

df['post'] = 0
post_mask = df.DST >= 0
df.loc[post_mask, 'post'] = 1

series_seq_dict = basic.df_to_dict(df_seq, SID, Seq)
df[Seq] = df.SID.map(series_seq_dict)


def all_equal(x): return len(set(x)) <= 1


gb_seq_pat = df.groupby(['Sequence', 'PID']).apply(
    lambda x: not(all_equal(x.post)))
df_pp = gb_seq_pat.reset_index()
df_pp.rename(columns={0: 'pre_post'}, inplace=True)
mask_ = df_pp.pre_post
df_pp = df_pp[mask_]
df_pp = df_pp.drop(columns=['pre_post'])
print(df_pp)

# In[Visualize]
seqs = df_pp.Sequence.unique()
series_pp_dict = {seq: df_pp[df_pp.Sequence == seq] for seq in seqs}
pat_pp_dict = {seq: series_pp_dict[seq].PID.unique() for seq in seqs}
pat_pp_counts = [len(df.PID.unique()) for df in list(series_pp_dict.values())]
svis.bar_plot(seqs, pat_pp_counts, dpi=100, save_plot=True,
              figname=f"{fig_dir}/patients_pre_post.png",
              title="# Patients with pre and post disease scan",
              xlabel='Sequence')

# In[Get patient lists]
        xlabel='Sequence')

# In[Get patient lists]





