# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 11:06:00 2021

@author: klein
"""
# %%
import pandas as pd
import matplotlib.pyplot as plt
import os, sys
from os.path import join, split
base_dir = split(os.getcwd())[0]
if base_dir not in sys.path:
    sys.path.append(base_dir)
import numpy as np
from stats_tools import vis as svis
from vis import mri_vis as mvis
import seaborn as sns
from utilities import stats, mri_stats
from utilities.basic import DotDict, p, sort_dict
import importlib
import pickle, gzip
import matplotlib as mpl
import matplotlib.lines as mlines
from collections import OrderedDict


from ast import literal_eval
plt.style.use(join(base_dir,'utilities', 'plot_style.txt'))
import importlib
import string
plt.style.use('ggplot')
#import proplot as pplt
import matplotlib.dates as mdates
import scipy.stats as ss

from pylab import cm
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
"""
params = {'figure.dpi':350,
        'legend.fontsize': 18,
        'figure.figsize': [8, 5],
         'axes.labelsize': 20,
         'axes.titlesize':20,
         'xtick.labelsize':20,
         'ytick.labelsize':20}
mpl.rcParams.update(params) 
plt.style.use('ggplot')
"""
#pplt.rc.cycle = 'ggplot'
colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] 

# %%
# In[tables directories]
fig_dir = f"{base_dir}/figs"
table_dir = f"{base_dir}/data/tables"
with gzip.open(join(table_dir, 'scan_3dt1_clean.gz'), 'rb') as f:
    df = pickle.load(f)
df['positive_scan'] = 0
df.loc[df.days_since_test>=-4, 'positive_scan'] = 1
df.PixelSpacing = df.PixelSpacing.map(lambda x: x.strip("[]").split(" "))
df = df[df.InstanceCreationDate!=df.InstanceCreationDate.min()]
df.InstanceCreationDate.hist()

# %%
df.SeriesInstanceUID.nunique()
#%%
# In[Counts]
print(len(df), df.PatientID.nunique())
df_pos = df[df.days_since_test>=-4]
pos_pats = df_pos.PatientID.unique() 
print(len(df_pos), len(pos_pats))

df_long_pats_case = df[df.PatientID.isin(pos_pats) & (df.days_since_test<=-30)]
long_cases = df_long_pats_case.PatientID.unique()
df_long_case = df[df.PatientID.isin(long_cases) & ( (df.days_since_test<=-30) | (df.days_since_test>=-4) ) ]
print(len(df_long_case),len(long_cases))
df_neg = df[((df.days_since_test<=-30)|df.days_since_test.isna()) & ~df.PatientID.isin(long_cases) ]
df_neg_gb = df_neg.groupby('PatientID').InstanceCreationDate.nunique()
neg_long_pats = df_neg_gb[df_neg_gb>=2].index.tolist()
df_cont_long = df[df.PatientID.isin(neg_long_pats) & ((df.days_since_test<=-30)|df.days_since_test.isna())]
print(len(df_cont_long), df_cont_long.PatientID.nunique())

df_cont_long_gb = df_cont_long.groupby('PatientID').SeriesInstanceUID.nunique()
len(df_cont_long_gb[df_cont_long_gb>=2].index.tolist())
#len(dfc_gb_long_pats)

#%%
# In[Cross-sec]
df_neg = df[~df.PatientID.isin(pos_pats) & ( (df.days_since_test<=-30) | (df.days_since_test.isna()) )]
print(len(df_neg), df_neg.PatientID.nunique())
# %%
# In[how many have scanner manufacturer, scanner type, b0 field strength]
TE_k = 'EchoTime'
TR_k = 'RepetitionTime'
TI_k = 'InversionTime'
FA_k = 'FlipAngle'
SD_k = 'SeriesDescription'
PID_k = 'PatientID'
time_k = 'InstanceCreationTime'
date_k = 'InstanceCreationDate'
DT_k = 'DateTime'
SID_k = 'SeriesInstanceUID'
SS_k = 'ScanningSequence'
SV_k = 'SequenceVariant'
SN_k = 'SequenceName'
SO_k = 'ScanOptions'
ETL_k = 'EchoTrainLength'
MFS_k = 'MagneticFieldStrength'
print(df.Manufacturer.isna().sum())
print(df.ManufacturerModelName.isna().sum())
print(df[MFS_k].isna().sum())
print(len(df))


# %%
# In[Plot MR field strength]
pos_mask = df.positive_scan==1
neg_value_counts = sort_dict(
    df[MFS_k][~pos_mask].value_counts())
pos_value_counts = sort_dict(
    df[MFS_k][pos_mask].value_counts())

# In[B0]
fig, ax = plt.subplots(1,2, figsize=(7,14))
ax = ax.flatten()
cmap = plt.get_cmap("Greys")
pcolor = cmap(np.array([0,50,150]))

labels_dic = {1.0:'1.0 T', 1.5:'1.5 T', 3.0:'3.0 T',}
new_labels = [labels_dic[l] for l in neg_value_counts.keys()]

def my_autopct(pct, th=6):
    return (f'{pct:.1f}%') if pct > th else ''
neg_data = list(neg_value_counts.values())
ts = ax[0].pie(neg_data, radius=1.1, labels=new_labels,
    wedgeprops=dict(width=.4, edgecolor=colors[1],linewidth=2), colors=pcolor,
    autopct=my_autopct, pctdistance=.84, labeldistance=1.14,
    textprops={'fontsize': 10})
for t in ts[1]:
    t.set_fontsize(15)
pos_data = list(pos_value_counts.values())
ax[0].pie(pos_data,labels=['']*3, radius=1-.5,
        wedgeprops=dict(width=.3, edgecolor=colors[0], linewidth=2), 
        colors=pcolor,autopct=my_autopct, pctdistance=.6, labeldistance=1.2,
        textprops={'fontsize': 10})

ax[0].set(aspect="equal")

pos_line = mlines.Line2D([], [], color=colors[0],  linestyle='-',
                          markersize=10, label='positive')
neg_line = mlines.Line2D([], [], color=colors[1],  linestyle='-',
                          markersize=10, label='negative')                          
ax[0].legend(handles=[pos_line, neg_line], loc=(.8,.9), fontsize=14)
#fig.tight_layout()
#fig.savefig(join(fig_dir, '3dt1','B0.png'),dpi=350)


# In[Plot Manufacturer]
pos_mask = df.positive_scan==1
neg_value_counts = sort_dict(
    df['Manufacturer'][~pos_mask].value_counts())
pos_value_counts = sort_dict(
    df['Manufacturer'][pos_mask].value_counts())


def sum_values(dic, neg=False):
    value_counts_dic = {}
    value_counts_dic['Siemens'] = dic['SIEMENS']
    if neg:
        value_counts_dic['Siemens'] = value_counts_dic['Siemens']+\
            dic['Siemens HealthCare GmbH']+dic['Siemens Healthineers']

    value_counts_dic['Philips'] = dic['Philips']+dic['Philips Healthcare']+\
        dic['Philips Medical Systems']
    value_counts_dic['others'] = dic['Agfa'] + dic['GE MEDICAL SYSTEMS']
    if neg:
        value_counts_dic['others'] = value_counts_dic['others'] + dic['Brainlab']
    return value_counts_dic
neg_value_counts = sum_values(neg_value_counts, neg=True)
pos_value_counts = sum_values(pos_value_counts)

print(neg_value_counts)

# In[Manufacturer]

#cmap = plt.get_cmap("Greys")
#pcolor = cmap(np.array([0,50,150]))

neg_data = list(neg_value_counts.values())
ts = ax[1].pie(neg_data, radius=1.1, labels=neg_value_counts.keys(),
    wedgeprops=dict(width=.4, edgecolor=colors[1],linewidth=2), colors=pcolor,
    autopct=my_autopct, pctdistance=.82, labeldistance=1.1,
    textprops={'fontsize': 10})
for t in ts[1]:
    t.set_fontsize(15)
pos_data = list(pos_value_counts.values())
ax[1].pie(pos_data, labels=['']*3, radius=1-.5,
        wedgeprops=dict(width=.3, edgecolor=colors[0], linewidth=2), 
        colors=pcolor, autopct=my_autopct, pctdistance=.7, labeldistance=1.2,
        textprops={'fontsize': 10}, startangle=50)

ax[1].set(aspect="equal")

pos_line = mlines.Line2D([], [], color=colors[0],  linestyle='-',
                          markersize=10, label='positive')
neg_line = mlines.Line2D([], [], color=colors[1],  linestyle='-',
                          markersize=10, label='negative')                 
for n, a in enumerate(ax):
    a.text(-0.1, 1.1, string.ascii_lowercase[n], transform=a.transAxes, 
            size=20, weight='bold')         
#ax[1].legend(handles=[pos_line, neg_line], loc=(.8,.9), fontsize=18)
fig.tight_layout()


fig.savefig(join(fig_dir, '3dt1','B0_manufacturer.png'),dpi=350, bbox_inches='tight')


# %%
# In[Same for all the different settings in the cross sectional study]
# for now draw random groups of patients
dfs = []
names = ['all', 'severe', 'short', 'long']
for i in range(4):
    df_imp = pd.read_csv(join(base_dir,'data','t1_crossec', 
        'groups_sids', names[i]+'.csv'))
    df_res = pd.merge(df_imp, df, on='SeriesInstanceUID', how='left')
    dfs.append(df_res)


#%%
titles = ['all', 'severe', 'short-term', 'long-term']
b0_labels = ['1.0 T', '1.5 T','3.0 T']
keys = neg_value_counts.keys()
fig, axs = plt.subplots(2,2, figsize=(5,5))
axs = axs.flatten()
startangles = [30,-30,30,20]
for i, ax in enumerate(axs):
    d = dfs[i]
    pval = ss.chi2_contingency(pd.crosstab(d[MFS_k], d.case))[1]
    #ax.annotate(r"$P=$"+str(round(pval,3)),(.3,1.1), fontsize=12 )
    pos_mask = d.case==1
    neg_value_counts = sort_dict(
        d[MFS_k][~pos_mask].value_counts())
    pos_value_counts = sort_dict(
        d[MFS_k][pos_mask].value_counts())
    pos_data = list(pos_value_counts.values())
    
 
    neg_data = list(neg_value_counts.values())
    if (len(pos_data)==2) & ~(1.0 in pos_value_counts.keys()):
        pos_data.insert(0,0)
    if (len(neg_data)==2) & ~(1.0 in neg_value_counts.keys()):
        neg_data.insert(0,0)
    
    ts = ax.pie(neg_data, radius=1.1, labels=['']*3,
        wedgeprops=dict(width=.4, edgecolor=colors[1],linewidth=2), colors=pcolor,
        autopct=my_autopct, pctdistance=.84, labeldistance=1.14,
        textprops={'fontsize': 10})
    ax.set_title(titles[i]+'  ('+r"$P=$"+str(round(pval,3))+')', loc='center')
    for t in ts[1]:
        t.set_fontsize(15)
    ax.pie(pos_data, labels=['']*3, radius=1-.4,
        wedgeprops=dict(width=.4, edgecolor=colors[0], linewidth=2), 
        colors=pcolor, autopct=my_autopct, pctdistance=.6, labeldistance=1.2,
        textprops={'fontsize': 10}, startangle=startangles[i])
    
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color=pcolor[0], lw=7),
                Line2D([0], [0], color=pcolor[1], lw=7),
                Line2D([0], [0], color=pcolor[2], lw=7),
                Line2D([0], [0], color=colors[0], lw=2),
                Line2D([0], [0], color=colors[1], lw=2),]

lgd = fig.legend(custom_lines, b0_labels+['cases','controls'], bbox_to_anchor=(0.5,0.12),
    borderpad=.8, ncol=2)
lgd.get_frame().set_facecolor('grey')
fig.savefig(join(fig_dir, 'crossec', 'B0.png'), dpi=300, bbox_inches='tight')
#%%
df.Manufacturer = df.Manufacturer.fillna('others')
df.Manufacturer.unique()
sim, phil, oth = 'Siemens', 'Philips', 'others'
manufacturer_dic = {'SIEMENS':sim, 'Siemens':sim, 
    'Siemens HealthCare GmbH':sim, 'Siemens Healthineers':sim,
    'Philips Healthcare':phil, 'Philips':phil, 
    'GE MEDICAL SYSTEMS':oth, 'Agfa':oth, 'nan':oth, 'Brainlab':oth,
    oth:oth}
df['Manufacturer_new'] = df.Manufacturer.map(manufacturer_dic)
dfs = []
names = ['all', 'severe', 'short', 'long']
for i in range(4):
    df_imp = pd.read_csv(join(base_dir,'data','t1_crossec', 
        'groups_sids', names[i]+'.csv'))
    df_res = pd.merge(df_imp, df, on='SeriesInstanceUID', how='left')
    dfs.append(df_res)
#%%
# In[Same for manufacturer]
fig, axs = plt.subplots(2,2, figsize=(5,5))
axs = axs.flatten()
man_labels = ['Siemens', 'Philips', 'others']
startangles_inner = [0,0,0,40]
startangles_outer = [20,0,20,20]
for i, ax in enumerate(axs):
    d = dfs[i]
    pval = ss.chi2_contingency(pd.crosstab(d['Manufacturer_new'], d.case))[1]
    #ax.annotate(r"$P=$"+str(round(pval,3)),(.3,1.1), fontsize=12 )
    pos_mask = d.case==1
    neg_value_counts = sort_dict(
        d['Manufacturer_new'][~pos_mask].value_counts())
    pos_value_counts = sort_dict(
        d['Manufacturer_new'][pos_mask].value_counts())
    pos_data = list(pos_value_counts.values())
    if (len(pos_data)==2) & ~(1.0 in pos_value_counts.keys()):
        pos_data.insert(0,0)
    neg_data = list(neg_value_counts.values())
    ts = ax.pie(neg_data, radius=1.1, labels=['']*3,
        wedgeprops=dict(width=.4, edgecolor=colors[1],linewidth=2), colors=pcolor,
        autopct=lambda x: my_autopct(x, 5), pctdistance=.82, labeldistance=1.14,
        textprops={'fontsize': 10}, startangle=startangles_outer[i])
    ax.set_title(titles[i]+'  ('+r"$P=$"+str(round(pval,3))+')', loc='center')
    for t in ts[1]:
        t.set_fontsize(15)
    ax.pie(pos_data, labels=['']*3, radius=1-.4,
        wedgeprops=dict(width=.4, edgecolor=colors[0], linewidth=2), 
        colors=pcolor, autopct=lambda x: my_autopct(x, 11), 
        pctdistance=.6, labeldistance=1.2,
        textprops={'fontsize': 10}, startangle=startangles_inner[i])
    
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color=pcolor[0], lw=7),
                Line2D([0], [0], color=pcolor[1], lw=7),
                Line2D([0], [0], color=pcolor[2], lw=7),
                Line2D([0], [0], color=colors[0], lw=2),
                Line2D([0], [0], color=colors[1], lw=2),]

lgd = fig.legend(custom_lines, man_labels+['cases','controls'], bbox_to_anchor=(0.5, 0.12),
    borderpad=.8, ncol=2)
lgd.get_frame().set_facecolor('grey')
fig.savefig(join(fig_dir, 'crossec', 'manufacturer.png'),
     dpi=300, bbox_inches='tight')





#%%
# In[Manufacturer Model Name]
titles = ['all', 'severe', 'short-term', 'long-term']
MMN_k = 'ManufacturerModelName'

for i in range(4):
    d = dfs[i]
    all_keys = d[MMN_k].value_counts().keys()
    top10_keys = all_keys[:10]
    mmn_dic = {k:k for k in top10_keys}
    mmn_dic_temp = {k:'others' for k in all_keys if k not in top10_keys}
    mmn_dic = {**mmn_dic, **mmn_dic_temp} 
    d['MMN_new'] = d[MMN_k].map(mmn_dic)
    dfs[i] = d
fig, axs = plt.subplots(2,2, figsize=(8,8))
axs = axs.flatten()
case_label='case'
control_label='control'
for i, ax in enumerate(axs):
    d = dfs[i]
    d_case = d[d.case==1]
    d_con = d[d.case==0]
    dic = d_con['MMN_new'].value_counts()/len(d_con)
    pval = ss.chi2_contingency(pd.crosstab(d['MMN_new'], d.case))[1]
    print(pval)
    # print(dic)
    names = list(dic.keys())
    values = np.array(list(dic.values))
    if i>0:
        control_label=''
    ax.barh(y = np.arange(0,22,2)+.25, width = values, height=.5, color=colors[1],
        label=control_label)
    ax.set_yticks(np.arange(0,22,2))
    ax.set_yticklabels(names)
    ax.tick_params(axis='y', which='major', labelsize=14)
    ax.set_ylim(-2,22)
    case_dic = d_case['MMN_new'].value_counts()/len(d_case)
    # create case_dic
    for k in dic.keys():
        if k in case_dic.keys():
            pass
        else:
            case_dic[k] = 0
    
    case_dic = OrderedDict([(el, case_dic[el]) for el in dic.keys()])
    case_names = list(case_dic.keys())
    case_values = np.array(list(case_dic.values()))
    if i>0:
        case_label=''
    ax.barh(y = np.arange(0,22,2)-.25, width = case_values, height=.5, color=colors[0],
        label=case_label)
    ax.set_title(titles[i], loc='center')
    ax.set_xlim(0,0.4)

fig.text(0.5, 0.04, r'$N/N_{\mathrm{tot}}$', ha='center', fontsize=14)
fig.text(-0.11, 0.5, 'Model Name', va='center', rotation='vertical', fontsize=14)
fig.legend( fontsize=14, bbox_to_anchor=(0.1, 1))
plt.subplots_adjust(wspace=.75, hspace=.3)
fig.savefig(join(fig_dir, 'crossec', 'model_name.png'),
     dpi=300, bbox_inches='tight')
#%%
fig, axs = plt.subplots(2,2, figsize=(5,5))
axs = axs.flatten()
for i, ax in enumerate(axs):
    d = dfs[i]
    #pval = ss.chi2_contingency(pd.crosstab(d[MMN_k], d.case))[1]
    #ax.annotate(r"$P=$"+str(round(pval,3)),(.3,1.1), fontsize=12 )
    pos_mask = d.case==1
    neg_value_counts = sort_dict(
        d[MMN_k][~pos_mask].value_counts())
    pos_value_counts = sort_dict(
        d[MMN_k][pos_mask].value_counts())
    pos_data = list(pos_value_counts.values())
    neg_data = list(neg_value_counts.values())
   
    d[d.case==0][MMN_k].value_counts().sort_values().plot(kind = 'barh', ax=ax)
    d[d.case==1][MMN_k].value_counts().sort_values().plot(kind = 'barh', ax=ax)
    
assert False
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color=pcolor[0], lw=7),
                Line2D([0], [0], color=pcolor[1], lw=7),
                Line2D([0], [0], color=pcolor[2], lw=7),
                Line2D([0], [0], color=colors[0], lw=2),
                Line2D([0], [0], color=colors[1], lw=2),]

lgd = fig.legend(custom_lines, b0_labels+['cases','controls'], bbox_to_anchor=(0.5,0.12),
    borderpad=.8, ncol=2)
lgd.get_frame().set_facecolor('grey')
fig.savefig(join(fig_dir, 'crossec', 'manufacturer_model_name.png'), dpi=300, bbox_inches='tight')


#%%
df = df.dropna(subset=['PixelSpacing','SpacingBetweenSlices'],
    axis=0)
df['RowSpacing'] = df.PixelSpacing.map(lambda x: float(x[0]))
df['ColumnSpacing'] = df.PixelSpacing.map(lambda x: float(x[1]))

#%%
# In[Rowspacing, Columnspacing, DBS]
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
dft = df.dropna(
    subset=['RowSpacing','ColumnSpacing','SpacingBetweenSlices'],axis=0)
ax.scatter(dft.RowSpacing, dft.ColumnSpacing, dft.SpacingBetweenSlices,
    s=1, )
ax.view_init(elev=30., azim=130)
ax.set_zlim(0,4)
ax.set_xlim(0,1.5)
ax.set_xlim(0,1)
ax.set_xlabel('RowSpacing')
ax.set_ylabel('ColumnSpacing')
ax.set_zlabel('SpacingBetweenSlices')
#%%
df.keys()

#%%
plot_vars = ['RowSpacing','ColumnSpacing','SpacingBetweenSlices', 'SliceThickness',
                'Rows','Columns','NumberOfSlices']
Q1 = df[plot_vars].quantile(0.25)
Q3 = df[plot_vars].quantile(0.75)
IQR = Q3 - Q1
df_temp = df[~((df[plot_vars] < (Q1 - 1.5 * IQR)) |(df[plot_vars] > (Q3 + 1.5 * IQR))).any(axis=1)]
color_dic = {0:colors[1], 1:colors[0]}
sns_plot = sns.pairplot(df_temp, 
    vars=plot_vars, 
      diag_kind="kde", hue='positive_scan', 
    plot_kws={"s": 13, 'alpha': 1}, palette=color_dic, corner=True,
    )

#sns_plot.savefig(f"{fig_dir}/3dt1/resolution_pairplot.png")
labels_dic = {"SliceThickness":r"$d_\mathrm{slice}$"+ " in mm", 
    "Rows":r'$N_\mathrm{rows}$', "Columns":r'$N_\mathrm{columns}$',
    "NumberOfSlices":r"$N_\mathrm{slices}$",
    'RowSpacing':r"$\Delta d_\mathrm{rows}$"+" in mm",
    "ColumnSpacing":r"$\Delta d_\mathrm{columns}$"+" in mm",
    "SpacingBetweenSlices":r"$\Delta d_\mathrm{slices}$"+" in mm", '':''}
num_plot_vars = len(plot_vars)
ax_mat = np.tril(np.reshape(np.arange(num_plot_vars**2), (num_plot_vars, num_plot_vars)))
ax_mat = ax_mat[ax_mat!=0]
#print(ax_mat)
# sns_plot.axes.flatten()[4].set_xlabel('asas')
for i, ax in enumerate(sns_plot.axes.flatten()):
    if i in ax_mat:
        ax.set_xlabel(labels_dic[ax.get_xlabel()], fontsize=20)
        ax.set_ylabel(labels_dic[ax.get_ylabel()], fontsize=20)
        ax.tick_params(labelsize=15)
handles = sns_plot._legend_data.values()
labels = sns_plot._legend_data.keys()
new_labels = ['positive', 'negative']
lgd_labels_dic = dict(zip(['1','0'],
    new_labels))
sns_plot._legend.remove()
plt.legend(handles=handles, labels=[lgd_labels_dic[l] for l in labels], 
    loc=(-.5,3.95), ncol=1, fontsize=20)
#sns_plot.tight_layout()
fig.savefig(join(fig_dir, '3dt1','B0.png'),dpi=400, bbox_inches='tight')
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

labels_dic = {'SliceThickness':"Slice Thickness in mm", 
    'RowSpacing':"Row Spacing in mm",'ColumnSpacing':"Column Spacing in mm",
    "SpacingBetweenSlices":"Spacing Between Slices in mm"}

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
# In[Get number of acquired volumes per patient]
scans_per_patient = df.groupby('PatientID').size()
figure = svis.hist(
    scans_per_patient, np.arange(1, 110, 2),
    show=True,kwargs={'xlabel':'# volumes per patient',
    'title':'All Patients'},
    save=True, figname=f"{fig_dir}/volumes_per_patient.png",
    )

#%%
scans_per_patient = df[df.Positive==0].groupby('PatientID').size()
figure = svis.hist(
    scans_per_patient, np.arange(1, 110, 2),
    show=True,kwargs={'xlabel':'# volumes per patient',
    'title':'Negative Patients'},
    save=True, figname=f"{fig_dir}/volumes_per_patient_neg.png",
    )

#%%
# In[Sort scans by manufacturer]
manufactureres = df['Manufacturer'].unique()
p(manufactureres)
philips_t = ['Philips Healthcare', 'Philips Medical Systems',
             'Philips']
philips_c = stats.check_tags(df, philips_t, 'Manufacturer').sum()
siemens_c = stats.mask_sequence_type(df, 'SIEMENS', 'Manufacturer').sum()
gms_c = stats.mask_sequence_type(
    df, 'GE MEDICAL SYSTEMS', 'Manufacturer').sum()
agfa_c = stats.mask_sequence_type(df, 'Agfa', 'Manufacturer').sum()
none_c = df['Manufacturer'].isnull().sum()
#%%
df.Manufacturer.unique()
#%%
# In[re]
# Replace manufacturers
df.Manufacturer[stats.check_tags(df, philips_t, 'Manufacturer')] = 'Philips'
df.Manufacturer[stats.check_tags(df,['Siemens'], 'Manufacturer')] = 'Siemens'
#%%
import matplotlib as mpl
# In[Plot sequence counts]
mpl.rcParams['figure.dpi'] = 400
plt.style.use('ggplot')
#labels = ['Siemens', 'Philips', 'GE', 'Other', 'other']
g = sns.countplot(x="Manufacturer", hue="Positive", data=df, hue_order=[1,0],
    order = df['Manufacturer'].value_counts().iloc[:3].index)
#g.set(xlabel=('Manufacturer'), ylabel=('Volume Count'), xticklabels=labels)
fig = g.get_figure()
fig.tight_layout()
fig.savefig(f"{fig_dir}/3dt1/manufacturer.png")
#%%
# In[visualize scanner manufacturer counts]
fig, ax = plt.subplots(1, figsize=(10, 6))
manufacturers_unq = ['Philips', 'SIEMENS', 'GE']
counts = np.array([philips_c, siemens_c, gms_c])
svis.bar(manufacturers_unq, counts, 
        kwargs={'xlabel':'Manufacturer', 'title':'All Patients'},
              save=True, 
              figname=f"{fig_dir}/manufacturers_count_all.png",
              )

#%%
# In[Model Name]
philips_m = stats.check_tags(df, philips_t, 'Manufacturer')
siemens_m = stats.mask_sequence_type(df, 'SIEMENS', 'Manufacturer')
gms_m = stats.mask_sequence_type(df, 'GE MEDICAL SYSTEMS', 'Manufacturer')

model_k = 'ManufacturerModelName'
philips_models_vc = df[philips_m][model_k].value_counts().to_dict()
siemens_models_vc = df[siemens_m][model_k].value_counts().to_dict()
gms_models_vc = df[gms_m][model_k].value_counts().to_dict()

# In[summarize small groups]
philips_models_vc_new = stats.group_small(philips_models_vc, 1000)
siemens_models_vc_new = stats.group_small(siemens_models_vc, 200)
gms_models_vc_new = stats.group_small(gms_models_vc, 200)

# In[visualize]
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
ax = ax.flatten()

lbls_ph = philips_models_vc_new.keys()
szs_ph = philips_models_vc_new.values()
lbls_si = siemens_models_vc_new.keys()
szs_si = siemens_models_vc_new.values()
lbls_gm = gms_models_vc_new.keys()
szs_gm = gms_models_vc_new.values()

ax[0].pie(szs_ph,  labels=lbls_ph, autopct='%1.1f%%',
          shadow=True, startangle=90)
# Equal aspect ratio ensures that pie is drawn as a circle.
ax[0].axis('equal')
ax[0].set_title('Philips', fontsize=20)
ax[1].pie(szs_si,  labels=lbls_si, autopct='%1.1f%%',
          shadow=True, startangle=90)
# Equal aspect ratio ensures that pie is drawn as a circle.
ax[1].axis('equal')
ax[1].set_title('Siemens', fontsize=20)
ax[2].pie(szs_gm,  labels=lbls_gm, autopct='%1.1f%%',
          shadow=True, startangle=90)
# Equal aspect ratio ensures that pie is drawn as a circle.
ax[2].axis('equal')
ax[2].set_title('GMS', fontsize=20)
ax[-1].axis('off')

fig.suptitle('Positive Patients', fontsize=20)
fig.tight_layout()
plt.subplots_adjust(wspace=.5, hspace=None)
plt.show()
fig.savefig(f"{fig_dir}/3dt1/model_name_pie_chart.png")

# In[Keys that are relevant]
rel_key_list = ['t1', 'gd', 't2', 't2s', 't2_flair', 'swi']


#%%

# In[Look at 'other' group] combine all the relevant masks to get others
seq_vars = [SD_k, TE_k, TR_k, FA_k, TI_k, ETL_k, SS_k, SV_k, SN_k]
ids_vars = [PID_k, SID_k]
comb_vars = seq_vars + ids_vars
nid_seq = df_p[mask_dict_p.none_nid]
nid_seq_sort = nid_seq[comb_vars].dropna(thresh=3).sort_values(by=SD_k,
                                                               axis=0,
                                                               ascending=True)
nid_seq_sort = nid_seq_sort.loc[nid_seq_sort.astype(
    str).drop_duplicates().index]

nid_seq_sort_ids = nid_seq_sort[ids_vars]
nid_seq_sort_ids.to_csv(f"{base_dir}/tables/non_identified_seq_ids.csv",
                        index=False)
nid_seq_sort[seq_vars].to_csv(f"{base_dir}/tables/non_identified_seq.csv",
                              index=False)
p(nid_seq_sort)


# In[Look at 'other' group for all mris]
mask_dict_all = mri_stats.get_masks_dict(df_all, False)
seq_vars = [SD_k, TE_k, TR_k, FA_k, TI_k,
            ETL_k, SS_k, SV_k, SN_k, PID_k, SID_k]

nid_seq = df_all[mask_dict_all.none_nid]
print(f"All not identified sequences {len(nid_seq)}")
nid_seq_sort = nid_seq[seq_vars].dropna(thresh=5).sort_values(by=SD_k,
                                                              axis=0,
                                                              ascending=True)
print(f"dropping missing columns {len(nid_seq_sort)}")
nid_seq_sort = nid_seq_sort.loc[nid_seq_sort.astype(
    str).drop_duplicates().index]
print(f"drop duplicates {len(nid_seq_sort)}")
nid_seq_sort = nid_seq_sort.drop_duplicates(subset=[SD_k])
print(f"drop some more duplicates {len(nid_seq_sort)}")
nid_seq_sort.to_csv(f"{base_dir}/tables/non_identified_seq_all.csv",
                    index=False)
# In[]
unique_sequence_names = df_all['SequenceName'].unique()
np.savetxt(f"{base_dir}/tables/unique_seq_names.txt",
           unique_sequence_names, fmt="%s")

# In[Show some scans]
mvis.show_series('a0be3bf699294420053eb3c6ca7d7f6c',
                '2e2caa22443d04a2d42bfc4e7dc9d6bb')
# p(nid_seq_sort)

# In[Save corresponding patient and scan ids]
ids_vars = [PID_k, SID_k]
nid_seq = df_p[mask_dict_p.none_nid]
nid_seq_sort = nid_seq[ids_vars].sort_values(by=PID_k, axis=0, ascending=True)
nid_seq_sort = nid_seq_sort.loc[nid_seq_sort.astype(
    str).drop_duplicates().index]
nid_seq_sort.to_csv(
    f"{base_dir}/tables/non_identified_seq_pid.csv", index=False)
p(nid_seq_sort)

# In[Get counts]
counts_dict = DotDict({key: mask.sum() for key, mask in mask_dict_p.items()})
print(counts_dict)

# In[visualize basic sequences]
sequences_names = ['T1+\nMPRAGE', 'FLAIR', 'T2', 'T2*', 'DWI', 'SWI',
                   'angio', 'ADC', 'Other', 'None or \n not identified']
seq_counts = np.array([counts_dict.t1, counts_dict.flair, counts_dict.t2,
                       counts_dict.t2s, counts_dict.dwi,
                       counts_dict.swi, counts_dict.angio, counts_dict.adc,
                       counts_dict.other, counts_dict.none_nid])
svis.bar(sequences_names, seq_counts, figsize=(13, 6), xlabel='Sequence',
              xtickparams_ls=16, save_plot=True, title='Positive Patients',
              figname=f"{fig_dir}/pos/basic_sequences_count.png")

# In[Visualize other sequences]
sequences_names = ['DTI', 'TRACEW', 'ASL', 'CEST', 'Survey', 'STIR',
                   'screensave', 'autosave']
seq_counts = np.array([counts_dict.dti, counts_dict.tracew, counts_dict.asl,
                       counts_dict.cest, counts_dict.survey,
                       counts_dict.stir,
                       counts_dict.screensave, counts_dict.autosave,
                       ])
svis.bar(sequences_names, seq_counts, figsize=(13, 6), xlabel='Sequence',
              xtickparams_ls=16, save_plot=True, title='Positive Patients',
              figname=f"{fig_dir}/pos/other_sequences_count.png")

# In[Look at the distributions of TE and TR for different seq]
df_p.loc[mask_dict_p.t1, 'Sequence'] = 'T1'
df_p.loc[mask_dict_p.t2, 'Sequence'] = 'T2'
df_p.loc[mask_dict_p.t2s, 'Sequence'] = 'T2S'
df_p.loc[mask_dict_p.flair, 'Sequence'] = 'FLAIR'
df_p_clean = df_p.dropna(subset=[TE_k, TR_k])

# In[visualize sequences scatter]
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
ax = ax.flatten()
sns.scatterplot(x=TE_k, y=TR_k,
                hue='Sequence', data=df_p_clean, ax=ax[0])
sns.scatterplot(x=TE_k, y=TI_k, legend=None,
                hue='Sequence', data=df_p_clean,
                ax=ax[1])
sns.scatterplot(x=TI_k, y=TR_k, legend=None,
                hue='Sequence', data=df_p_clean,
                ax=ax[2])
sns.scatterplot(x=TI_k, y=FA_k, legend=None,
                hue='Sequence', data=df_p_clean,
                ax=ax[3])
fig.suptitle('Identified Sequences (positive patients)', fontsize=20)
fig.tight_layout()
plt.show()
fig.savefig(f"{fig_dir}/pos/scatter_training.png")

# In[Extract dates from series description if not present in InstanceCreationData]
# p(df_p['InstanceCreationDate'].dropna())
p(f"number of scans without date {df_p['InstanceCreationDate'].isnull().sum()}\
  out of {len(df_p)}")
date_mask = df_p['SeriesDescription'].str.contains('2020', na=False)
# p(df_p[date_mask]['SeriesDescription'].count())
# these are not that many

# In[Search for combinations of FLAIR, SWI, T1]
gb_pat = df_p.groupby(PID_k)
# grouped masks followd by
flair_m = stats.check_tags(gb_pat, tag_dict_p.flair).groupby(PID_k).any()
swi_m = stats.check_tags(gb_pat, tag_dict_p.swi).groupby(PID_k).any()
t1_m = stats.check_tags(gb_pat, tag_dict_p.t1).groupby(PID_k).any()
t2_m = stats.check_tags(gb_pat, tag_dict_p.t2).groupby(PID_k).any()

flair_swi_t1_m = flair_m & swi_m & t1_m
p(f"{flair_swi_t1_m.sum()} patients have\
  the sequences flair, swi and t1")
flair_swi_t1_t2_m = flair_m & swi_m & t1_m & t2_m
p(f"{flair_swi_t1_t2_m.sum()} patients have\
  the sequences flair, swi,t1 and t2")


# In[Number of studies per month/year]

ps_datetime_count = df_p.groupby(
    [df_p[DT_k].dt.year, df_p[DT_k].dt.month]).count()[SID_k]
year_month_keys = [str(int(key[1]))+'/'+str(key[0])[:4]
                   for key in ps_datetime_count.keys()]
year_month_keys.insert(-1, '5/2021')  # this month is missing
year_month_counts = ps_datetime_count.values
year_month_counts = np.insert(year_month_counts, -1, 0)
svis.bar(year_month_keys[:-3], year_month_counts[:-3], figsize=(13, 7),
              xtickparams_rot=70,
              xlabel='month/year', save_plot=(True), ylabel='Frequency',
              title='Number of acquired volumes for positive patients',
              figname=f"{fig_dir}/pos/scans_months_years.png")

# In[when is the date present but not a time]
p(f"{pd.isnull(df_p[date_k]).sum()} scans dont have a time or date")

# In[Study months distribution]
importlib.reload(stats)
_, study_dates = stats.time_between_studies(df_p)

# In[Studies distr]
year_month_study_dates = [str(date.year)+'/'+str(date.month)
                          for date in study_dates]
year_month_unique, year_month_counts = np.unique(
    np.array(year_month_study_dates), return_counts=True)
svis.bar(year_month_unique[:-2], year_month_counts[:-2], figsize=(13, 7),
              xtickparams_rot=70, xlabel='study month/year', save_plot=(True),
              ylabel='Frequency', title='Studies for positive patients',
              figname=f"{fig_dir}/pos/studies_months_years.png")




# In[Find intersection of FLAIR, DWI, SWI or T2*]
comb0 = ['dwi', 'flair', 'swi',]
comb1 = comb0 + ['t1']
comb2 = ['dwi', 'flair', 't2s_gre',]
comb3 = comb2 + ['t1']
comb_list = [comb0, comb1, comb2, comb3]
comb_labels = ['dwi, flair, swi', 'dwi, flair, swi\n +t1',
          'dwi, flair, t2* gre', 'dwi, flair, t2* gre\n +t1',]
dfp = df_all[df_all.Pos==1]
dfn = df_all[df_all.Pos==0]
Pos_combs_list = []
Neg_combs_list = []

for comb in comb_list:
    pos_setlist = []
    neg_setlist = []
    for tag in comb:
        pos_setlist.append(set(dfp[dfp.Sequence==tag].PatientID.unique()))
        neg_setlist.append(set(dfn[dfn.Sequence==tag].PatientID.unique()))
    pos_intersection = len(set.intersection(*pos_setlist))
    neg_intersection = len(set.intersection(*neg_setlist))
    Pos_combs_list.append(pos_intersection)
    Neg_combs_list.append(neg_intersection)
    
fig, ax = svis.bar(comb_labels, Neg_combs_list, figsize=(14,6), width=.6, 
                   label='neg')
fig, ax = svis.bar(comb_labels, Pos_combs_list, label='pos', bottom=Neg_combs_list, 
         width=.6, color=(0,1), fig=fig, ax=ax, 
         kwargs={'xlabel':'Sequence Type Combinations', 
                 'ylabel':'Patient Count', 'yrange':(0,7000)},
         ) 
                  
for i in range(4):
    ax.text(i-.1, Pos_combs_list[i]+Neg_combs_list[i]+200, 
            Pos_combs_list[i]+Neg_combs_list[i], fontsize=20)
    ax.text(i+.35, Neg_combs_list[i]+50, Pos_combs_list[i],
            color=svis.Color_palette(0)[1],
            fontsize=20, ha='left')
    
fig.savefig(fname=f"{fig_dir}/basic_stats/sequence_comb_pat_count.png", dpi=100)
# In[Write to files]
write_dir = f"{base_dir}/share/Cerebriu/download_patients"
with open(f"{write_dir}/dwi_flair_t2s.txt", 'w') as f:
    for item in dwi_flair_t2s:
        f.write("%s\n" % item)
# In[Count number of relevant patients in 2019]
print(len(df_2019[mask_dict.t1 | mask_dict.t2 | mask_dict.t2s | mask_dict.t1gd
                  | mask_dict.t2gd | mask_dict.gd | mask_dict.swi | mask_dict.flair
                  | mask_dict.dwi][PID_k].unique()))
print(len(df_2019[PID_k].unique()))
