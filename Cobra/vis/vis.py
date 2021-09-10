import matplotlib.pyplot as plt
import math
import numpy as np


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

##########################################
def bar_plot(labels, counts, figsize=(10,6), width=.8,
             lgd_label='', lgd=False, lgd_loc=0, lgd_fs=25,
             ticklabels_fs=25, title='', title_fs=25,
             ylabel='count', ylabel_fs=25,
             xlabel='x', xlabel_fs=25,
             tickparams_ls=25, logscale=False,
             save_plot=False, figname=None, dpi=80):
    fig, ax = plt.subplots(1,figsize = figsize)
    ax.bar(np.arange(len(counts)), counts, width, label = lgd_label)
    x = np.arange(len(counts))  # the label locations
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=ticklabels_fs)
    if lgd:
        ax.legend(loc=lgd_loc, fontsize=lgd_fs, facecolor = 'white')
    if title!='':
        ax.set_title(title, fontsize=title_fs)
    ax.set_ylabel(ylabel, fontsize=ylabel_fs)
    ax.set_xlabel(xlabel, fontsize=xlabel_fs)
    ax.tick_params(axis='both', which='major', labelsize=tickparams_ls)
    if logscale:
        ax.set_yscale('log')
    fig.tight_layout()
    if save_plot:
        fig.savefig(figname, dpi=dpi)

