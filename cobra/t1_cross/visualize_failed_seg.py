#%%
from matplotlib import gridspec
from matplotlib.pyplot import suptitle
import nibabel as nib
import sys
import os
from os.path import split, join
base_dir = split(os.getcwd())[0]
if base_dir not in sys.path:
    sys.path.append(base_dir)
from utilities import basic
nib.Nifti1Header.quaternion_threshold = -1e-06
import proplot as pplt
import matplotlib.pyplot as plt
import numpy as np
from nilearn import plotting

#%%
# In[Load data]
pred_dir0 = "F:\\CoBra\\Data\\volume_cross_nii\\prediction"
pred_dir1 = "F:\\CoBra\\Data\\volume_longitudinal_nii\\prediction"
input_dir0 = "F:\\CoBra\\Data\\volume_longitudinal_nii\\input\\nii_files\\segmented"
input_dir1 = "F:\\CoBra\\Data\\volume_cross_nii\\input\\nii_files\\segmented"
input_dir2 = "F:\\CoBra\\Data\\volume_cross_nii\\input\\nii_files"
inputs = basic.list_subdir(input_dir0, ending='.nii.gz')+\
    basic.list_subdir(input_dir1, ending='.nii.gz')+\
        basic.list_subdir(input_dir2, ending='.nii.gz')
segs = basic.list_subdir(pred_dir0, ending='_seg.nii.gz')+\
    basic.list_subdir(pred_dir1, ending='_seg.nii.gz')
show_ids = ['001622', '006709', '010238']

show_inputs = [file for file in inputs if \
    (split(file)[1]).split('.nii')[0] in show_ids]
show_segs = [file for file in segs if \

    (split(file)[1]).split('_seg')[0] in show_ids]


#%%
fig, axs = plt.subplots(ncols=1, nrows=3, figsize=(13,10))
axs = axs.flatten()
cut_coords_ls = [(-24,14,10), (39,1,-40)]
for i, show_inp in enumerate(show_inputs):
    x = nib.load(show_inp)
    y = nib.load(show_segs[i])
    d = plotting.plot_img(x, axes=axs[i],figure=fig, cut_coords=cut_coords_ls[i],
        cmap='gray', draw_cross=False)
    d.add_overlay(y, cmap='roy_big_bl', threshold=0)
plt.subplots_adjust(hspace=.01)
fig.tight_layout()
#%%
fig, ax = plt.subplots()
d = plotting.plot_img(show_inputs[2], figure=fig, axes=ax,cut_coords=(39,1,0),cmap='gray',
    draw_cross=False)
fig.tight_layout()
#d.add_overlay(show_segs[2], cmap='videen_style', threshold=0)
#%%
#%%
fig, axs = pplt.subplots(ncols=5, nrows=2, share=False,refaspect=.9, hspace=('3pt'),
    wspace=('1pt','1pt','1pt','1pt',))

axs[0].imshow(x_ring_c[40:-30,185,:], cmap='gray',aspect=.6)
axs[1].imshow(x_ring_c[:,158,:], cmap='gray')
axs[2].imshow(x_ghost_c[20:-20,158,:], cmap='gray')

axs[3].imshow(x_ring_c[55:-55,185,:], cmap='gray',aspect=.54)
axs[4].imshow(x_ring_c[:,158,:], cmap='gray')

# Augmented
axs[5].imshow(x_org[:,120,:].T, cmap='gray')
axs[6].imshow(x_bf[:,120,:].T, cmap='gray',)
axs[7].imshow(x_deform_1[:,120,:].T-x_org[:,120,:].T, cmap='gray',)
axs[8].imshow(x_ring_1[:,120,:].T, cmap='gray')
axs[9].imshow(x_ghost_1[:,120,:].T, cmap='gray',vmax=x_org[:,120,:].max()-2300)
#CPH
for i, ax in enumerate(axs):
    if i in np.arange(5,10):
        ax.set_xlim(0, 210)
        ax.set_ylim(0, 250)
axs.format(
    leftlabels=('Copenhagen', 'MICCAI   \n Augmented'),#
    toplabels=('None','Bias Field', 'Deformation', 'Ringing', 'Ghosting'),
    xticks=[],
    yticks=[],
    yticklabels=[],
    xticklabels=[],
    xtickminor=False, ygridminor=False,
)
#%%
fig, axs = pplt.subplots(ncols=5, nrows=1, share=False,
    wspace=('1pt','1pt','1pt','1pt',))

# Augmented
axs[0].imshow(x_org[:,120,:].T, cmap='gray')
axs[1].imshow(x_bf[:,120,:].T, cmap='gray',)
axs[2].imshow(x_deform_1[:,120,:].T-x_org[:,120,:].T, cmap='gray',)
axs[3].imshow(x_ring_1[:,120,:].T, cmap='gray')
axs[4].imshow(x_ghost_1[:,120,:].T, cmap='gray',vmax=x_org[:,120,:].max()-2300)

axs.format(
    toplabels=('Original','Bias Field', 'Original-Deformation', 'Ringing', 'Ghosting'),
    xticks=[],
    yticks=[],
    xlim=(0, 210),
    ylim=(0,250),
    yticklabels=[],
    xticklabels=[],
    xtickminor=False, ygridminor=False,
)
fig.savefig("C:\\Users\\kiril\\Thesis\\CoBra\\cobra\\figs\\augmentations\\augmentations_000_strong.pdf")
