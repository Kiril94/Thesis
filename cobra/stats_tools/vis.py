# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 15:18:15 2021

@author: klein
"""
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from scipy import stats
from . import fits
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker
import seaborn as sns
from stats_tools.colors import Color_palette
import os, sys
from os.path import split, join
base_dir = split(os.getcwd())[0]
if base_dir not in sys.path:
    sys.path.append(base_dir)
plt.style.use(join(base_dir,'utilities', 'plot_style.txt'))
def quickplot(x, kind='scatter', y=None, 
        xlabel='', ylabel='', title=''):
    fig, ax = plt.subplots()
    if y is not None:
        if kind=='scatter':
            ax.scatter(x, y)
        elif kind=='plot':
            ax.plot(x,y)
        else:
            print('kind ', kind,'not implemented')
            return 1
    else:
        ax.hist(x)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_title(title, fontsize=22)
    fig. tight_layout()



# In[Helper]
def ax_decorator(fig, ax,
                 lgd_loc=0, lgd_fs=25, 
                 lgd_color='white', lgd_ncol=1,
                 title='', title_fs=25, 
                 ylabel='', ylabel_fs=25,
                 xlabel='', xlabel_fs=25,
                 xticks=[], xtick_labels=[],
                 yticks=[], ytick_labels=[],
                 xtickparams_ls=25, xtickparams_rot=0,
                 ytickparams_ls=25, ytickparams_rot=0,
                 xlogscale=False, ylogscale=False,
                 xrange=(None,None), yrange=(None,None),
                 tight_layout=True,plot_style=None,
                 **kwargs):
    if not isinstance(plot_style, type(None)):
        plt.style.use(plot_style)
    _, labels = ax.get_legend_handles_labels()
    if len(labels)>0:
        legend = ax.legend(loc=lgd_loc, fontsize=lgd_fs, facecolor='white',
                           ncol=lgd_ncol)
        legend.get_frame().set_facecolor(lgd_color)
    ax.set_title(title, fontsize=title_fs)
    ax.set_ylabel(ylabel, fontsize=ylabel_fs)
    ax.set_xlabel(xlabel, fontsize=xlabel_fs)
    ax.tick_params(axis='x', which='major', labelsize=xtickparams_ls,
                   rotation=xtickparams_rot)
    ax.tick_params(axis='y', which='major', labelsize=ytickparams_ls,
                   rotation=ytickparams_rot)
    if len(xticks)!=0:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels)
    if len(xticks)!=0:
        ax.set_yticks(yticks)
        ax.set_yticklabels(ytick_labels)
    ax.set_xlim(xrange)
    ax.set_ylim(yrange, auto=True)
    if xlogscale:
        ax.set_yscale('log')
    if ylogscale:
        ax.set_yscale('log') 
    if tight_layout:
        fig.tight_layout()
    return fig, ax


# In[General plots]

def plot_decorator(plot_func, plot_func_args=[], plot_func_kwargs={},
                   figsize=(8, 5), save=False, dpi=80, figname='',
                   caption='', caption_fs=20, caption_pos=(.5, -.05),
                   plot_style=None,
                   kwargs={}):
    """Takes a function plot_func which takes args, 
    kwargs and ax to produce a plot"""
    if not isinstance(plot_style, type(None)):
        plt.style.use(plot_style)
    fig, ax = plt.subplots(figsize=figsize)
    ax = plot_func(*plot_func_args, **plot_func_kwargs, ax=ax)
    try:
        fig, ax = ax_decorator(fig, ax, **kwargs)
    except:
        if save:
            ax.figure.savefig(figname, dpi=dpi)
            save=False
        pass
    if caption != '':
        fig.text(caption_pos[0], caption_pos[1], caption, ha='center',
                 wrap=True, fontsize=caption_fs)  
    if save:
        fig.savefig(figname, dpi=dpi)
    return fig, ax


def line(
        x, y, sy=0, errorbar=False, scatter=False, absolute_sigma=True,
        show=True, save=False, figname=None, 
        label='', figsize=(10, 5), 
        ax=None, fig=None, text_fs=14,
        dpi=80, color='skyblue',
        plot_style=None, linestyle='solid', ecolor='deepskyblue',
        capsize=3, capthick=0.3, err_markersize=6,  elinewidth=.9,
        alpha=1, scr_markersize=30, scr_markerstyle='o', linewidth=3,
        fill_color=None, drawstyle='default',
        kwargs={}):
    r"""
    Simple x-y plot. 

    Parameters:
    ----------

    x, y, sy: of array_like,
        input data, if produced from histogram, pass ONLY values where Y>0
    errorbar: bool, False by default, if True errorbars are plotted 
    
    Returns:
    -------
    fig: figure object
    ax: axis object

    """

    if ax is None or fig is None:
        fig, ax = plt.subplots(figsize=figsize)
    try:
        line_color = Color_palette(color[0])[color[1]]
    except:
        line_color = color
    if type(ecolor) == tuple:
        ecolor = Color_palette(ecolor[0])[ecolor[1]]
    if type(fill_color) == tuple:
        fill_color = Color_palette(fill_color[0])[fill_color[1]]
    if not isinstance(plot_style, type(None)):
        plt.style.use(plot_style)
    const_err = (type(sy) == float)
    if const_err:
        sy = np.ones(len(x))*sy
    if errorbar:
        ax.errorbar(x, y, yerr=sy, marker='.', mec=ecolor, color=ecolor,
                    elinewidth=elinewidth, capsize=capsize, capthick=capthick,
                    linestyle='none', markersize=err_markersize,
                    label=label, alpha=alpha)  # plot data
    elif scatter:
        ax.scatter(
            x, y, color=line_color,  label=label,
            linestyle=linestyle, s=scr_markersize,
            alpha=alpha, marker=scr_markerstyle)
    else:
        ax.plot(x, y, color=line_color,  label=label, drawstyle=drawstyle,
                linestyle=linestyle, alpha=alpha, linewidth=linewidth)
        
    if fill_color != None:
        plt.fill_between(x, y, color=fill_color)
    fig, ax = ax_decorator(fig, ax, **kwargs)
    if save:
        fig.savefig(figname, dpi=dpi)

    return fig, ax

##########################################

def bar(labels, counts, width=.8,
        label='', figsize=(10, 6), save=False,
        figname=None, dpi=80, plot_style=None,
        fig=None, ax=None, bottom=[], 
        caption='', caption_fs=20, caption_pos=(.5, -.05),
        color = '',
        kwargs={}):
    if not isinstance(plot_style, type(None)):
        plt.style.use(plot_style)
   
    if ax is None or fig is None:
        fig, ax = plt.subplots(figsize=figsize)
    if len(bottom) == 0:
        bottom = np.zeros(len(counts))
    if color is '':
        color = Color_palette(0)[0]
    ax.bar(np.arange(len(counts)), counts, width, label=label,
           bottom=bottom, color=color)
    x = np.arange(len(counts))  # the label locations
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    fig, ax = ax_decorator(fig, ax, **kwargs)
    if save:
        fig.savefig(figname, dpi=dpi)
    return fig, ax

#########################


def hist(
        x_all, N_bins, poisson_error=False, xlogscale=False,
        show=False, plot_hist=True, xrange=None,
        save=False, figname='',
        data_label='Data, histogram', data_label_hist='',
        figsize=(12, 6), histtype='step', color_hist='red', 
        ax=None, fig=None,
        dpi=80, ecolor='deepskyblue', capsize=3, capthick=0.3,
        markersize=6, elinewidth=.9, hist_alpha=.9, hist_linestyle='solid',
        hist_linewidth=2, plot_style=None, 
        kwargs={}):
    """Produce a nice histogram.
    Returns: dictionary with x, y, sy, binwidth, fig, ax."""
    if not(xrange is None):
        mask_x = (x_all > xrange[0]) & (x_all < xrange[1])
        x_all = x_all[mask_x]

    if poisson_error:
        x, y, sy, binwidth = fits.produce_hist_values(
            x_all, N_bins, x_range=xrange,
            log=xlogscale, poisson_error=poisson_error)
    else:
        x, y, binwidth = fits.produce_hist_values(
            x_all, N_bins, x_range=xrange,
            log=xlogscale, poisson_error=poisson_error)

    if type(color_hist) == tuple:
        color_hist = Color_palette(color_hist[0])[color_hist[1]]
    if type(ecolor) == tuple:
        ecolor = Color_palette(ecolor[0])[ecolor[1]]

    if ax is None or fig is None:
        fig, ax = plt.subplots(figsize=figsize)
    if not isinstance(plot_style, type(None)):
        plt.style.use(plot_style)
    

    if plot_hist:
        ax.hist(x_all, bins=N_bins, range=(x_all.min(), x_all.max()),
                histtype=histtype, linewidth=hist_linewidth, color=color_hist,
                label=data_label_hist, alpha=hist_alpha,
                linestyle=hist_linestyle)
    if poisson_error:
        ax.errorbar(
            x, y, yerr=sy, xerr=0.0, label=data_label, marker='.',
            mec=ecolor, color=ecolor, elinewidth=elinewidth,
            capsize=capsize, capthick=capthick, linestyle='none',
            markersize=markersize)

    fig, ax = ax_decorator(fig, ax, **kwargs)     
    if save:
        fig.savefig(figname, dpi=dpi)
    if show:
        plt.show()
    Figure = {"x": x, "y": y, "binwidth": binwidth, "fig": fig, "ax": ax}
    if poisson_error:
        Figure["sy"] = sy

    return Figure

#############################


def contour(
        xx, yy, z, levels=40, cmap='inferno', colors=None, figsize=(12, 6),
        filled=True, fig=None, ax=None, plot_style=None,
        show_cbar=True, cbar_size='5%', cbar_pad=0.1, label_fs=20,
        tick_size=20, cbar_tick_labelsize=20,  cbar_orientation='vertical',
        cbar_label='', cbar_label_fs=20, cbar_num_ticks=5, clabels=None,
        labels_inline=True, clabel_fs=18, plot_clabels=False,
        linewidths=1.5, linestyles='solid',
        show=True, save=False, figname='',
        dpi=80, kwargs={}):
    """Producing nice contour plot. 
    Parameters:
        xx, yy, zz: xx and yy are meshgrids, z is a 2d matrix
    Returns:
        fig, ax
    """

    if ax is None or fig is None:
        fig, ax = plt.subplots(figsize=figsize)

    if not isinstance(plot_style, type(None)):
        plt.style.use(plot_style)
    if colors != None:
        cmap = None

    if filled:
        im = ax.contourf(xx, yy, z, levels=levels, cmap=cmap)
    else:
        im = ax.contour(
            xx, yy, z, levels=levels, cmap=cmap, colors=colors,
            linestyles=linestyles, linewidths=linewidths)
        if plot_clabels:
            ax.clabel(im, inline=labels_inline, fontsize=clabel_fs)
        for i in range(len(clabels)):
            im.collections[i].set_label(clabels[i])
    
    # set axes params
    fig, ax = ax_decorator(fig, ax, **kwargs)

    # colorbar
    if show_cbar:
        divider = make_axes_locatable(ax)
        if cbar_orientation == 'vertical':
            cbar_loc = 'right'
        else:
            cbar_loc = 'top'

        cax = divider.append_axes(cbar_loc, size=cbar_size, pad=cbar_pad)
        cbar = fig.colorbar(im, cax=cax, orientation=cbar_orientation)
        cbar.ax.tick_params(labelsize=cbar_tick_labelsize)
        if cbar_orientation == 'vertical':
            cbar.ax.set_ylabel(cbar_label, fontsize=cbar_label_fs)
        else:
            cbar.ax.set_title(cbar_label, fontsize=cbar_label_fs, color=ax)
            cax.xaxis.set_ticks_position('top')
        tick_locator = ticker.MaxNLocator(nbins=cbar_num_ticks)
        cbar.locator = tick_locator
        cbar.update_ticks()

    if save:
        fig.savefig(figname, dpi=dpi)

    if show:
        plt.show()

    return fig, ax
#######################################


def scatter_hist(X0, X1, Y0, Y1, ax, ax_histx, ax_histy, N_bins_x, N_bins_y,
                 histlabel0, histlabel1):
    """Helper function to create additional axes with histograms."""

    # set tick parameters
    ax_histx.tick_params(axis="x", labelbottom=False,)
    ax_histy.tick_params(axis="y", labelleft=False)
    ax_histx.tick_params(axis="y", labelsize=18)
    ax_histy.tick_params(axis="x", labelsize=18)

    # x ranges for histograms
    x_min = np.amin([X0, X1])
    x_max = np.amax([X0, X1])
    y_min = np.amin([Y0, Y1])
    y_max = np.amax([Y0, Y1])

    # create actual histograms
    ax_histx.hist(X0, bins=N_bins_x, range=(x_min, x_max),
                  alpha=.8, label=histlabel0)
    ax_histx.hist(X1, bins=N_bins_x, range=(x_min, x_max),
                  alpha=.8, label=histlabel1)
    ax_histy.hist(Y0, bins=N_bins_y, range=(y_min, y_max),
                  orientation='horizontal', alpha=.8)
    ax_histy.hist(Y1, bins=N_bins_y, range=(y_min, y_max),
                  orientation='horizontal', alpha=.8)
    ax_histx.legend(fontsize=12)

###########################################


def plot_classification(X, y, classifier, N_bins_x=40, N_bins_y=40,
                        label0='type I', label1='type II',
                        histlabel0='type I', histlabel1='type II',
                        xlabel='x', ylabel='y',
                        figsize=(10, 10), save=False, figname='',
                        show_plot=False):
    r"""
    Create a scatter plot including separation according to classifier #
    with histograms of projections on x and y axis.

    Parameters:
    -----------
    X: array_like
        Input data with N rows and 2 columns, where N is the number of samples
        and 2 variables
    y: array_like
        Target data with N rows and 1 column, contains 0s and 1s
    classifier: class from sklearn
    label0, label1: str, optional,
        labels for first and second variables, default H0,H1
    xlabel, ylabel: str, optional, default x, y
    save: bool, default(False)
    figname: if save==True 
        saves figure under figname
    N_bins_x, N_bins_y: int 
        specifies number of bins for first and second variable
    label0, label1, histlabel0, histlabel1: str, optional
    xlabel, ylabel: str, optional
    figsize: (float,float), optional, default (10,10)
    show: bool, show plot if True

    Returns:
    -------
    classifier: object, fitted classifier that contains 
        fit parameters and can be used to make predictions
    ax_scatter, ax_histx, ax_histy: axis objects, can be used to add a fit
        or change parameters
    fig: fig object, can be used for saving when sth changed
    """

    classifier.fit(X, y)

    fig, ax = plt.subplots()  # create figure
    scatter_kwargs = {'s': 100, 'edgecolor': 'k', 'alpha': 1,
                      'marker': 's'}
    contourf_kwargs = {'alpha': .3}
    # plot decision boundaries
    ax_scatter = plot_decision_regions(X=X, y=y,
                                       clf=classifier, legend=2,
                                       scatter_kwargs=scatter_kwargs,
                                       contourf_kwargs=contourf_kwargs,
                                       ax=ax)
    ax_scatter.set_xlim(X[:, 0].min()*1.1, X[:, 0].max()*1.1)
    ax_scatter.set_ylim(X[:, 1].min()*1.1, X[:, 1].max()*1.1)
    ax_scatter.tick_params(labelsize=20)
    ax_scatter.set_xlabel(xlabel, fontsize=20)
    ax_scatter.set_ylabel(ylabel, fontsize=20)
    handles, labels = ax_scatter.get_legend_handles_labels()  # get pos of axis
    bbox = ax_scatter.get_position()
    left, bottom, width, height = bbox.bounds
    spacing = 0.04
    ax_scatter.legend(handles, [label0, label1],
                      framealpha=0.3, scatterpoints=1, fontsize=20,
                      bbox_to_anchor=(left+width, bottom+height, 0.5, 0.5))
    # set position for additional axes
    rect_histx = [left, bottom + height + spacing, width, 0.25]
    rect_histy = [left + width + spacing, bottom, 0.25, height]

    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)
    mask0 = y == 0
    X0 = X[mask0, 0]
    X1 = X[~mask0, 0]
    Y0 = X[mask0, 1]
    Y1 = X[~mask0, 1]
    scatter_hist(X0, X1, Y0, Y1, ax_scatter, ax_histx, ax_histy,
                 N_bins_x=N_bins_x, N_bins_y=N_bins_y,
                 histlabel0=histlabel0, histlabel1=histlabel1)

    if save:
        fig.savefig(figname, bbox_inches='tight')

    if show_plot:
        plt.show()

    return classifier, ax_scatter, ax_histx, ax_histy, fig

###########################


def add_zoom_inset(ax, zoom, loc, x, y, xlim, ylim, sy=None,
                   xlabel='', ylabel='', label_fs=18,
                   mark_inset_loc=(3, 1), borderpad=4):
    """Add inset axis that shows a region of the data.

    Parameters:
    -----------
    ax: axis object
    zoom: float, zoom factor
    loc: integer, location of inset axis
    x,y, sy: array_like, data to plot
    xlim, ylim: (float, float) limits for x and y axis
    xlabel, ylabel: str, label for x and y axis
    label_fs: float, fonstsize for x and y axis labels
    mark_inset_loc: (int, int), corners for connection to the new axis
    borderpad: float, distance from border
    """

    axins = zoomed_inset_axes(ax, zoom, loc=loc,
                              borderpad=borderpad)
    if sy is None:
        axins.plot(x, y)
    else:
        axins.errorbar(x, y, yerr=sy, fmt='.b',  ecolor='b', elinewidth=.5,
                       capsize=0, capthick=0.1)

    if not(ylim is None):
        axins.set_ylim(ylim)

    axins.set_xlabel(xlabel, fontsize=label_fs)
    axins.set_ylabel(ylabel, fontsize=label_fs)
    axins.set_xlim(xlim)  # Limit the region for zoom
    mark_inset(ax, axins, loc1=mark_inset_loc[0],
               loc2=mark_inset_loc[1], fc="none", ec="0.5")


# In[Random Numbers]

############################
def create_1d_hist(ax, values, bins, x_range, title, histtype='stepfilled',display_counts = False):
    """Helper function for show_int_distribution. (Author: Troels Petersen)"""
    ax.hist(values, bins, x_range, histtype=histtype, density=False, lw=2)
    ax.set(xlim=x_range, title=title)
    hist_data = np.histogram(values, bins, x_range)
    
    bin_width = hist_data[1][1] - hist_data[1][0]
    if (display_counts):
        for i in range(bins):
            if (hist_data[0][i]!=0):
                ax.text(hist_data[1][i]+bin_width/4,hist_data[0][i],str(hist_data[0][i]))

    return hist_data

################################

def create_boxplot(data, data_labels=None,title='',fig=None, ax=None,
                    plot_style=None, figsize=(10, 10), kwargs={},
                    stripplot=False, orient='v'):
    if ax is None or fig is None:
        fig, ax = plt.subplots(figsize=figsize)
    if not isinstance(plot_style, type(None)):
        plt.style.use(plot_style)
    if (data_labels is None):
        data_labels = [int(i+1) for i in range(len(data))]
    ax = sns.boxplot(data=data,palette='Set2', medianprops=dict(color="red"), ax=ax,
        orient=orient)
    if stripplot:
        ax = sns.stripplot(data=data, color=".25",alpha=0.3, ax=ax)
    if orient=='v':
        ax.set_xticklabels(data_labels)
    else:
        ax.set_yticklabels(data_labels)
    ax.get_xaxis().tick_bottom()
    ax.set_title(title)
    fig, ax = ax_decorator(fig, ax, **kwargs)
    return fig, ax


def get_chi2_ndf(hist, const):
    """Helper function for show_int_distribution. (Author: Troels Petersen)"""
    data = hist[0]
    const_unnormed = const * data.sum()
    chi2 = np.sum((data - const_unnormed)**2 / data)
    ndof = data.size
    return chi2, ndof

#################################


def show_int_distribution(integers, save_plot=True,
                          figname='', show_plot=False):
    """Show histogram of integers, to see if random.(Author: Troels Petersen)
    modified by: Kiril Klein
    Parameters: 
        integers, array_like
    Returns: 
        dict_raw, dict_odd_even, dict_high_low: dictionaries
            contains chi2, ndf, p for the hypothesis of integers being random.
        AX: list of axes objects
        fig: figure
        """
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    ax_number, ax_odd_even, ax_high_low = ax.flatten()

    # Fill 1d histograms and plot them:
    hist_numbers = create_1d_hist(ax_number, integers, 10,
                                  (-0.5, 9.5), 'Numbers posted')  # Plot all digits
    hist_odd_even = create_1d_hist(ax_odd_even, integers % 2,  2,
                                   (-0.5, 1.5), 'Even and odd numbers')  # Is number even or odd
    hist_high_low = create_1d_hist(ax_high_low, integers // 5, 2,
                                   (-0.5, 1.5), 'Above and equal to or below 5')  # Is number >= or < 5
    fig.tight_layout()

    chi2_raw, ndf_raw = get_chi2_ndf(hist_numbers,  1.0 / 10)
    chi2_odd_even, ndf_odd_even = get_chi2_ndf(hist_odd_even, 1.0 / 2)
    chi2_high_low, ndf_high_low = get_chi2_ndf(hist_high_low, 1.0 / 2)
    p_raw = stats.chi2.sf(chi2_raw, ndf_raw)
    p_odd_even = stats.chi2.sf(chi2_odd_even, ndf_odd_even)
    p_high_low = stats.chi2.sf(chi2_high_low, ndf_high_low)

    dict_raw = {'chi2': chi2_raw, 'ndf': ndf_raw, 'p': p_raw}
    dict_odd_even = {'chi2': chi2_odd_even,
                     'ndf': ndf_odd_even, 'p': p_odd_even}
    dict_high_low = {'chi2': chi2_high_low,
                     'ndf': ndf_high_low, 'p': p_high_low}
    AX = [ax_number, ax_odd_even, ax_high_low]
    if save_plot:
        fig.savefig(figname)
    if show_plot:
        plt.show()

    return dict_raw, dict_odd_even, dict_high_low, fig, AX

####################################################
def create_2d_hist(ax,x_values,y_values,bins,range_values=None,title=''):
    h,xedges,yedges,image = ax.hist2d(x_values,y_values,range=range_values,bins=bins)
    ax.set_title(title)
    plt.colorbar(image,ax=ax)
    return h,xedges,yedges

# In[Helper functions]

##################################
def resolution_boxplot(df, title='', save=False, figname=''):
    fig, ax = plt.subplots(1,4, figsize=(22,8),gridspec_kw={'width_ratios': [2, 1,2,2]})
    fig, _ = create_boxplot(df[['Rows', 'Columns',]], 
        data_labels=["Height", "Width", ], 
        kwargs={'xlabel':'',},
        fig=fig, ax=ax[0])
    create_boxplot(df[['NumberOfSlices', ]], 
        data_labels=["Depth", ], 
        kwargs={'xlabel':'','yrange':(0,500)},
        fig=fig, ax=ax[1])
    create_boxplot(df[['RowSpacing', 'ColumnSpacing',]], 
        data_labels=["Pixel \nSpacing x", "Pixel \nSpacing y",], 
        kwargs={'xlabel':'','ylabel':'mm','yrange':(0,2)},
        fig=fig, ax=ax[2])
    create_boxplot(df[[ 'SpacingBetweenSlices','SliceThickness' ]], 
        data_labels=["Slices\nSpacing","Slice\nThickness" ], 
        kwargs={'xlabel':'','ylabel':'mm','yrange':(0,7)},
        fig=fig, ax=ax[3])
    fig.suptitle(title, fontsize=25)
    fig.tight_layout()
    if save:
        fig.savefig(figname)




# In[Example]
"""
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy import stats
from sklearn.metrics import plot_roc_curve

clf = LinearDiscriminantAnalysis(solver='svd')

rng = np.random.RandomState(0)
x1v1 = np.random.normal(-1,.2,200)
x2v1 = np.random.normal(0,.5,200)
x1v2 = np.random.normal(-1,.2,200)
x2v2 = np.random.normal(0,.5,200)

y1 = np.zeros(200, dtype = 'int')
y2 = np.ones(200, dtype = 'int')
X1 = np.concatenate((x1v1[:,np.newaxis],x1v2[:,np.newaxis]), axis = 1)
X2 = np.concatenate((x2v1[:,np.newaxis],x2v2[:,np.newaxis]), axis = 1)
X = np.concatenate((X1,X2))
y = np.concatenate((y1,y2))
fitted_clf,_,ax_histx, ax_histy, fig = plot_classification(
        X,y, clf, save = True, figname = 'test2.png', show = False)
ax_histx.plot(np.linspace(-2,2,100), 10*stats.norm.pdf(np.linspace(-2,2,100), 
                                                      loc = 0,scale = .5))
display(fig)
plot_roc_curve(fitted_clf, X, y)
"""
