#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 11:06:16 2021

@author: neus
"""

import numpy as np
import pandas as pd
import itertools

import matplotlib.pyplot as plt


import sys 
import vis as svis

sys.path.append('../vis/')
import vis


sys.path.append('../utilss/')
import stats
main_path = '/home/neus/Documents/09.UCPH/MasterThesis/github/Thesis/Cobra/'
file_name = 'healthy_1'


tab_dir = main_path+'tables/'
fig_dir = main_path+'figs/basic_stats'

# In[Define some helper functions]

def p(x):
    print(x)

def only_first_true(a, b):
    """takes two binary arrays
    and returns True
    where only the el. of the first array is true
    if only the second or both are true returns false"""
    return a&np.logical_not(a&b)

def mask_sequence_type(df, str_, key='SeriesDescription'):
    """Checks all the values in df
    in the column key (SeriesDescription by default),
    if they contain the string str_. Returns a mask."""
    mask = df[key].str.contains(str_, na=False)
    return mask

def check_tags(df, tags, key='SeriesDescription'):
    """calls mask_sequence type for a list of tags and combines
    the masks with or"""
    masks = []
    for tag in tags:
        masks.append(mask_sequence_type(df, tag, key))
    mask = masks[0]
    for i in range(1, len(masks)):
        mask = mask | masks[i]
    return mask

def group_small(dict_, threshold, keyword='other'):
    """Takes a dictionary and sums all values that are smaller than threshold
    the result is stored under the key keyword. Useful for pie charts."""
    newdic={}
    for key, group in itertools.groupby(dict_, lambda k: keyword \
                                        if (dict_[k]<threshold) else k):
         newdic[key] = sum([dict_[k] for k in list(group)]) 
    return newdic

def get_property(prop,column_names,data):
    '''
    
    Parameters
    ----------
    prop : str
        DESCRIPTION.
    column_names : np array
        keys of the data.
    data : np array
        whole array of data.

    Returns
    -------
    np array
        np array for the given property.

    '''
    index = np.where(column_names==prop)[0]
    return data[:,index]

def read_table(file_name,encoding= 'unicode_escape'):
    '''
    Parameters
    ----------
    file_name : str
        table file csv name.
    encoding : str, optional
        encoding . The default is 'unicode_escape'.

    Returns
    -------
    array for column names, np array for data, sorted by acquisition time.

    '''
    df = pd.read_csv(file_name,encoding=encoding)
    
    
    #In[Convert time and date to datetime for efficient access]
    time_k = 'InstanceCreationTime'
    date_k = 'InstanceCreationDate'
    df['DateTime'] = df[date_k] + ' ' +  df[time_k]
    #date_time_m = df['DateTime'].isnull()
    df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y%m%d %H:%M:%S')

    #In[Sort the the scans by time]
    df_sorted = df.groupby('PatientID').apply(
        lambda x: (x.sort_values(by=['DateTime'], ascending=True)))
        
    return df_sorted.keys(), df_sorted.to_numpy()

def count_studies_per_patient(column_names,data):
    df = pd.DataFrame(data=data,columns=column_names)
    
    df = pd.DataFrame(data=data,columns=column_names)
    unique_studies = df.drop_duplicates(subset='StudyInstanceUID')
    studies_per_patient = unique_studies.groupby('PatientID').size()
        
    return studies_per_patient

def plot_n_studies(column_names,data,fig_dir=fig_dir,file_name=file_name,fig_name='num_studies'):
    count_studies = count_studies_per_patient(column_names,data)
    num_studies_a = count_studies.values
    figure = svis.nice_histogram(num_studies_a, np.arange(-.5, max(num_studies_a)+.5),
                        show_plot=True, xlabel='Number of studies',
                        save=True, 
                        figname=f"{fig_dir}/{file_name}/{fig_name}.png")
    return figure
    
def plot_n_scans(column_names,data,fig_dir=fig_dir,file_name=file_name,fig_name='num_scans'):
    df = pd.DataFrame(data=data,columns=column_names)
    scans_per_patient = df.groupby('PatientID').size()
    figure = svis.nice_histogram(
    scans_per_patient, np.arange(1,110,2), 
    show_plot=True, xlabel='# volums per patient',
    save=True, figname = f"{fig_dir}/{file_name}/{fig_name}.png")
    return figure 

def sort_by_manufacturer(column_names,data):
    df = pd.DataFrame(data=data,columns=column_names)
    
    manufactureres = df['Manufacturer'].unique()
    p(manufactureres)
    philips_t = ['Philips Healthcare', 'Philips Medical Systems',
                 'Philips'] 
    philips_c = check_tags(df, philips_t, 'Manufacturer').sum()
    siemens_c = mask_sequence_type(df, 'SIEMENS', 'Manufacturer').sum()
    gms_c = mask_sequence_type(df, 'GE MEDICAL SYSTEMS', 'Manufacturer').sum()
    agfa_c = mask_sequence_type(df, 'Agfa', 'Manufacturer').sum()
    none_c = df['Manufacturer'].isnull().sum()
    
    manufacturers_unq = ['Philips', 'SIEMENS', 'GEMS', 'Agfa', 'none']
    counts = np.array([philips_c, siemens_c, gms_c, agfa_c, none_c])
    
    return manufacturers_unq,counts

def plot_manufacturers(column_names,data,fig_dir=fig_dir,file_name=file_name,fig_name='manufacturers_count'):
    manufacturers_unq,counts = sort_by_manufacturer(column_names,data)
    figure = vis.bar_plot(manufacturers_unq, counts, xlabel='Manufacturer', 
             save_plot=True, figname=f"{fig_dir}/{file_name}/{fig_name}.png")
    
    return figure

def plot_models(column_names,data,fig_dir=fig_dir,file_name=file_name,fig_name='model_name_pie_chart'):
    df = pd.DataFrame(data=data,columns=column_names)
    
    philips_t = ['Philips Healthcare', 'Philips Medical Systems',
             'Philips'] 
    
    philips_m = check_tags(df, philips_t, 'Manufacturer')
    siemens_m = mask_sequence_type(df, 'SIEMENS', 'Manufacturer')
    gms_m = mask_sequence_type(df, 'GE MEDICAL SYSTEMS', 'Manufacturer')
    
    model_k = 'ManufacturerModelName'
    philips_models_vc = df[philips_m][model_k].value_counts().to_dict()
    siemens_models_vc = df[siemens_m][model_k].value_counts().to_dict()
    gms_models_vc = df[gms_m][model_k].value_counts().to_dict()
    
    #In[summarize small groups]
    philips_models_vc_new = group_small(philips_models_vc, 1000)
    siemens_models_vc_new = group_small(siemens_models_vc, 200)
    gms_models_vc_new = group_small(gms_models_vc, 200)
    
    #In[visualize]
    fig, ax = plt.subplots(2,2, figsize=(10,10))
    ax = ax.flatten()
    
    lbls_ph = philips_models_vc_new.keys()
    szs_ph = philips_models_vc_new.values()
    lbls_si = siemens_models_vc_new.keys()
    szs_si = siemens_models_vc_new.values()
    lbls_gm = gms_models_vc_new.keys()
    szs_gm = gms_models_vc_new.values()
    
    ax[0].pie(szs_ph,  labels=lbls_ph, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax[0].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax[0].set_title('Philips', fontsize=20)
    ax[1].pie(szs_si,  labels=lbls_si, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax[1].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax[1].set_title('Siemens', fontsize=20)
    ax[2].pie(szs_gm,  labels=lbls_gm, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax[2].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax[2].set_title('GMS', fontsize=20)
    ax[-1].axis('off')
    
    fig.tight_layout()
    plt.subplots_adjust(wspace=.5, hspace=None)
    plt.show()
    fig.savefig(f"{fig_dir}/{file_name}/{fig_name}.png")


    return fig

def group_by_seq(data,column_names):
    
    df = pd.DataFrame(data=data,columns=column_names)
    tag_dict = {}
    tag_dict['t1'] = ['T1', 't1']
    tag_dict['mpr'] = ['mprage', 'MPRAGE']
    tag_dict['tfe'] = ['tfe', 'TFE']
    tag_dict['spgr'] = ['FSPGR']
    tag_dict['smartbrain'] = ['SmartBrain']
    
    tag_dict['flair'] = ['FLAIR','flair', 'Flair']
    
    tag_dict['t2'] = ['T2', 't2']
    tag_dict['fse'] = ['FSE', 'fse', '']
    
    tag_dict['t2s'] = ['T2\*', 't2\*']
    tag_dict['gre']  = ['GRE', 'gre']
    
    tag_dict['dti']= ['DTI', 'dti']
    tag_dict['swi'] = ['SWI', 'swi']
    tag_dict['dwi'] = ['DWI', 'dwi']
    tag_dict['adc'] = ['ADC', 'Apparent Diffusion Coefficient']
    tag_dict['gd'] = ['dotarem', 'Dotarem', 'Gd','gd', 'GD', 'Gadolinium']
    tag_dict['stir'] = ['STIR']
    tag_dict['tracew'] = ['TRACEW']
    tag_dict['asl'] = ['ASL']
    tag_dict['cest'] = ['CEST']
    tag_dict['survey'] = ['SURVEY', 'Survey', 'survey']
    tag_dict['angio'] = ['TOF', 'ToF', 'tof','angio', 'Angio', 'ANGIO', 'SWAN']
    tag_dict = DotDict(tag_dict)
    # Look up: MIP (maximum intensity projection), SmartBrain, 
    # TOF (time of flight angriography), ADC?, STIR (Short Tau Inversion Recovery),
    # angio, Dynamic Contrast-Enhanced Magnetic Resonance Imaging (DCE-MRI) 
    # In[Get corresponding masks]
    # take mprage to the t1
    mask_dict = DotDict({key : stats.check_tags(df, tag) for key, tag in tag_dict.items()})
    #mprage is always t1 https://pubmed.ncbi.nlm.nih.gov/1535892/
    mask_dict['t1'] = stats.check_tags(df, tag_dict.t1) \
        | stats.check_tags(df, tag_dict.mpr)
    mask_dict['t1tfe'] = mask_dict.t1 & mask_dict.tfe
    mask_dict['t1spgr'] = mask_dict.t1 & mask_dict.spgr
    mask_dict['t2_flair'] = stats.only_first_true(
        stats.check_tags(df, tag_dict.t2), mask_dict.t2s)
    mask_dict['t2_noflair'] = stats.only_first_true(mask_dict.t2_flair, mask_dict.flair)# real t2
    mask_dict.none = df['SeriesDescription'].isnull()
    
    mask_dict.all = mask_dict.t1 | mask_dict.flair | mask_dict.t2_noflair \
        | mask_dict.t2s | mask_dict.dwi | mask_dict.dti | mask_dict.swi \
            | mask_dict.angio | mask_dict.adc | mask_dict.stir \
                |mask_dict.survey | mask_dict.none
    mask_dict.other = ~mask_dict.all
    # In[Look at 'other' group] combine all the relevant masks to get others
    
    p(df[mask_dict.other].SeriesDescription)
    other_seq_series = df[mask_dict.other].SeriesDescription
    other_seq_series_sort = other_seq_series.sort_values(axis=0, ascending=True).unique()
    pd.DataFrame(other_seq_series_sort).to_csv(f"{tab_dir}/other_sequences.csv")
    
    
    # In[Get counts]
    counts_dict = DotDict({key : mask.sum() for key, mask in mask_dict.items()})
    
    # In[visualize basic sequences]
    sequences_names = ['T1+\nMPRAGE', 'T2', 'FLAIR', 'T2*', 'SWI', 'DWI', 
                       'angio', 'ADC', 'survey','TRACEW', 'Other','None']
    seq_counts = np.array([counts_dict.t1, counts_dict.t2_noflair, counts_dict.flair,
                           counts_dict.t2s, counts_dict.swi, 
                           counts_dict.dwi, counts_dict.angio, 
                           counts_dict.adc, counts_dict.survey, counts_dict.tracew,
                           counts_dict.other, 
                           counts_dict.none])
    vis.bar_plot(sequences_names, seq_counts, figsize=(13,6), xlabel='Sequence',
                 xtickparams_ls=16, save_plot=True, title='Positive Patients',
                 figname=f"{fig_dir}/{file_name}/{fig_name}.png")

    return sequences_names,seq_counts

def plot_seq(column_names,data,fig_dir=fig_dir,file_name=file_name,fig_name='basic_sequences_count'):
    sequences_basic, seq_counts = group_by_seq(data,column_names)
    figure = vis.bar_plot(sequences_basic, seq_counts, figsize=(13,6), xlabel='Sequence',
             xtickparams_ls=18, save_plot=True, 
             figname=f"{fig_dir}/{file_name}/{fig_name}.png")
    
    return figure





# In[Read table]
#column_names,data = read_table(tab_dir+file_name+'.csv')

