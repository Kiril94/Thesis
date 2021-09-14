#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 11:06:16 2021

@author: neus
"""

import numpy as np
import pandas as pd
import itertools
from stats_tools import vis as svis
from vis import vis
import matplotlib.pyplot as plt

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
    #n[Sequence Types]
    t1_t = ['T1', 't1']
    mpr_t = ['mprage', 'MPRAGE']
    tfe_t = ['tfe', 'TFE']
    spgr_t = ['FSPGR']
    smartbrain_t = ['SmartBrain']
    
    flair_t = ['FLAIR','flair']
    
    t2_t = ['T2', 't2']
    fse_t = ['FSE', 'fse', '']
    
    t2s_t = ['T2\*', 't2\*']
    gre_t  = ['GRE', 'gre']
    
    dti_t = ['DTI', 'dti']
    
    swi_t = ['SWI', 'swi']
    dwi_t = ['DWI', 'dwi']
    gd_t = ['dotarem', 'Dotarem', 'Gd','gd', 'GD', 'Gadolinium']
    
    angio_t = ['TOF', 'ToF', 'angio', 'Angio', 'ANGIO']
    # Look up: MIP (maximum intensity projection), SmartBrain, 
    # TOF (time of flight angriography), ADC?, STIR (Short Tau Inversion Recovery),
    # angio, Dynamic Contrast-Enhanced Magnetic Resonance Imaging (DCE-MRI) 
    #In[Get corresponding masks]
    # take mprage to the t1
    t1_m = check_tags(df, t1_t) | check_tags(df, mpr_t)
    
    mpr_m = check_tags(df, mpr_t)
    t1mpr_m = t1_m & mpr_m
    tfe_m = check_tags(df, tfe_t)
    t1tfe_m = t1_m & tfe_m
    spgr_m = check_tags(df, spgr_t)
    t1spgr_m = t1_m & spgr_m
    
    flair_m = check_tags(df, flair_t)
    
    fse_m = check_tags(df, fse_t)
    
    t2s_m = check_tags(df, t2s_t)
    gre_m  = check_tags(df, gre_t)
    
    dwi_m = check_tags(df, dwi_t)
    gd_m = check_tags(df, gd_t)
    
    t2_flair_m = only_first_true(check_tags(df, t2_t), t2s_m)
    t2_noflair_m = only_first_true(t2_flair_m, flair_m)# real t2
    dti_m = check_tags(df, dti_t)
    
    swi_m = check_tags(df, swi_t) 
    
    angio_m = check_tags(df, angio_t)
    smartbrain_m  = check_tags(df, smartbrain_t)
    
    none_m = df['SeriesDescription'].isnull()
    # we are interested in t1, t2_noflair, flair, swi, dwi, dti
    # combine all masks with an or and take complement
    all_m = t1_m | flair_m | t2_noflair_m | t2s_m | dwi_m | dti_m | swi_m | angio_m | none_m 
    other_m = ~all_m
    
    #In[Get counts]

    t1mpr_c = t1mpr_m.sum()
    t1tfe_c = t1tfe_m.sum()
    t1spgr_c = t1spgr_m.sum()
    
    # counts we are interested in
    flair_c = flair_m.sum()
    t1_c = t1_m.sum()
    t2_c = t2_flair_m.sum()
    t2noflair_c = t2_noflair_m.sum()
    t2s_c = t2s_m.sum()
    dti_c = dti_m.sum()
    swi_c = swi_m.sum()
    dwi_c = dwi_m.sum()
    angio_c = angio_m.sum()
    
    none_c = none_m.sum()
    other_c = other_m.sum()
    
    sequences_basic = ['T1+MPR', 'T2', 'FLAIR', 'T2*', 'DTI', 'SWI', 'DWI', 'angio',
                   'Other',
                   'None']
    seq_counts = np.array([t1_c, t2noflair_c, flair_c, t2s_c, 
                       dti_c, swi_c, dwi_c, angio_c, other_c, none_c])

    return sequences_basic,seq_counts

def plot_seq(column_names,data,fig_dir=fig_dir,file_name=file_name,fig_name='basic_sequences_count'):
    sequences_basic, seq_counts = group_by_seq(data,column_names)
    figure = vis.bar_plot(sequences_basic, seq_counts, figsize=(13,6), xlabel='Sequence',
             xtickparams_ls=18, save_plot=True, 
             figname=f"{fig_dir}/{file_name}/{fig_name}.png")
    
    return figure





# In[Read table]
#column_names,data = read_table(tab_dir+file_name+'.csv')

