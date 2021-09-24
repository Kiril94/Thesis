import matplotlib.pyplot as plt
import math
import numpy as np
from data_access import load_data_tools as ld
import glob


def display3d(arr3d, figsize=(15, 15), start_slice=0, num_slices=None, axis=0):
    """Input: 3D array of shape (num_slices, h, w)
    Returns: fig, ax
    Displays all the slices of a 3d array. 
    You can also display the slices along axis 1 or 2 by setting the axis argument."""
    
    if num_slices==None:
        num_slices = len(arr3d)
    if num_slices>50:
        num_slices = 50
    data_shape = arr3d.shape
    if data_shape[axis]<(start_slice+num_slices):
        print("The data has only {data_shape[axis]} slices.")
        start_slice = 0
        num_slices = data_shape[axis]
    cols = 5
    rows = math.floor(num_slices/cols)
    figsize = (figsize[0], math.floor(rows/cols*figsize[0]))
    fig, ax = plt.subplots(rows, cols, figsize=figsize)
    ax = ax.flatten()
    plt.axis('off')
    if axis==1:
        arr3d = np.transpose(arr3d, [1,0,2])
    elif axis==2:
        arr3d = np.transpose(arr3d, [2,0,1])
    else:
        pass
    for i, a in enumerate(ax):
        a.imshow(arr3d[i+start_slice], cmap='gray')
        a.axis('off')
        a.set_title(f"slice = {i+start_slice}")
    fig.tight_layout()
    return fig, ax

def show_series(patient_id, series_id, base_data_dir='Y:'):
    """Shows the series for a series_id"""
    counter = 0
    for series_dir in glob.iglob(f"{base_data_dir}/*/{patient_id}/*/*/{series_id}"):
        if counter>0:
            break
        arr3d = ld.reconstruct3d(series_dir)
        display3d(arr3d)
        counter += 1
        
        
##########################################
def bar_plot(labels, counts, figsize=(10,6), width=.8,
             lgd_label='', lgd=False, lgd_loc=0, lgd_fs=25,
             title='', title_fs=25,
             ylabel='count', ylabel_fs=25,
             xlabel='x', xlabel_fs=25,
             xtickparams_ls=25, xtickparams_rot=0, ytickparams_ls=25, logscale=False,
             save_plot=False, figname=None, dpi=80, plot_style='ggplot'):
    fig, ax = plt.subplots(1,figsize = figsize)
    plt.style.use(plot_style)
    ax.bar(np.arange(len(counts)), counts, width, label = lgd_label)
    x = np.arange(len(counts))  # the label locations
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    if lgd:
        ax.legend(loc=lgd_loc, fontsize=lgd_fs, facecolor = 'white')
    if title!='':
        ax.set_title(title, fontsize=title_fs)
    ax.set_ylabel(ylabel, fontsize=ylabel_fs)
    ax.set_xlabel(xlabel, fontsize=xlabel_fs)
    ax.tick_params(axis='x', which='major', labelsize=xtickparams_ls,
                   rotation=xtickparams_rot)
    ax.tick_params(axis='y', which='major', labelsize=ytickparams_ls)
    if logscale:
        ax.set_yscale('log')
    fig.tight_layout()
    if save_plot:
        fig.savefig(figname, dpi=dpi)

##############################################################
def plot_decorator(plot_func, args, kwargs, 
                   figsize=(9,9), save=False, dpi=80, figname='',
                   lgd=False, lgd_loc=0, lgd_fs=25, lgd_color='white', 
                   lgd_ncol=1, lgd_shadow=True,
                   set_xlabel=False, xlabel='count', xlabel_fs=25,
                   set_ylabel=False, ylabel='count', ylabel_fs=25,
                   set_xticks=False, xticks=[], xtick_labels=[],
                   set_yticks=False, yticks=[], ytick_labels=[],
                   set_xtickparams=False, xtickparams_ls=25, xtickparams_rot=0, 
                   set_ytickparams=False, ytickparams_ls=25, ytickparams_rot=0,
                   xlogscale=False, ylogscale=False,):
    
    """Takes a function plot_func which takes args, 
    kwargs and ax to produce a plot"""
    fig, ax = plt.subplots(figsize=figsize)
    ax = plot_func(*args, **kwargs, ax=ax)
    
    if lgd:
        ax.legend(loc=lgd_loc, fontsize=lgd_fs, facecolor=lgd_color,
                  ncol=lgd_ncol, shadow=lgd_shadow)
    if set_ylabel:
        ax.set_ylabel(ylabel, fontsize=ylabel_fs)
    if set_xlabel:
        ax.set_xlabel(xlabel, fontsize=ylabel_fs)
    if set_xticks:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels)
    if set_yticks:
        ax.set_yticks(yticks)
        ax.set_yticklabels(ytick_labels)
    if set_xtickparams:
        ax.tick_params(axis='x', which='major', labelsize=xtickparams_ls,
                       rotation=xtickparams_rot)
    if set_ytickparams:
        ax.tick_params(axis='y', which='major', labelsize=ytickparams_ls,
                       rotation=ytickparams_rot)
    if xlogscale:
        ax.set_xscale('log')
    if ylogscale:
        ax.set_yscale('log')
        
    fig.tight_layout()
    if save:
        fig.savefig(figname, dpi=dpi)
        
        
        
        
        
        
        
        
        
        
        