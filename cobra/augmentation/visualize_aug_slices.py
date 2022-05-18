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
nib.Nifti1Header.quaternion_threshold = -1e-06
import proplot as pplt
import numpy as np


#%%
cop_dir = "F:\\CoBra\\Data\\volume_cross_nii\\input\\nii_files\\segmented"
aug_dir = "C:\\Users\\kiril\\Thesis\\CoBra\\cobra\\data\\augmentations\\MICCAI\\examples"
org_dir = "C:\\Users\\kiril\\Thesis\\CoBra\\cobra\\data\\augmentations\\MICCAI\\imagesTr"

def load_arr(id,which=None, aug=None):
    if which=='c':
        print('Copenhagen')
        return nib.load(join(cop_dir,  id+'.nii.gz')).get_fdata()
    elif which=='o':
        return nib.load(join(org_dir, 'MICCAI_0'+ id+'_0000'+'.nii.gz')).get_fdata()
    else:
        return nib.load(join(
            aug_dir, 'MICCAI_0'+ id+'_0000_'+str(aug)+'_'+str(which)+'.nii.gz')
            ).get_fdata()
#%%
ids_ringing = []
ids_ghosting = []
ids_deform = []
ids_bf = []
#%%
x_org = load_arr('00', which='o')
x_bf = load_arr('00', which='1', aug=1)
#x_deform_0 = load_arr('00', which='0', aug=2)
x_deform_1 = load_arr('00', which='1', aug=2)
#x_ring_0 = load_arr('00', which='0', aug=3)
x_ring_1 = load_arr('00', which='1', aug=3)
x_ring_c = load_arr('000450', which='c')
#x_ghost_0 = load_arr('00', which='0', aug=4)
x_ghost_1 = load_arr('00', which='1', aug=4)
x_ghost_c = load_arr('000129', which='c')
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
#%%
gs = pplt.GridSpec(nrows=2, ncols=3, pad=0)
fig = pplt.figure(span=False, refaspect=.45, wspace=('0pt','0pt',))
pplt.rc['abc.size']=15
ax = fig.subplot(gs[:, 0])
ax.set_title('Original')
ax.imshow(x_org[:,120,:].T, cmap='gray')

ax = fig.subplot(gs[0, 1])
ax.set_title('Bias Field')
ax.imshow(x_bf[:,120,:].T, cmap='gray')

ax = fig.subplot(gs[0, 2])
ax.set_title('Original-Deformation')
ax.imshow(x_deform_1[:,120,:].T-x_org[:,120,:].T, cmap='gray')

ax = fig.subplot(gs[1, 1])
ax.set_title('Ringing')
ax.imshow(x_ring_1[:,120,:].T, cmap='gray')

ax = fig.subplot(gs[1, 2])
ax.set_title('Ghosting')
ax.imshow(x_ghost_1[:,120,:].T, cmap='gray',vmax=x_org[:,120,:].max()-2300)#
fig.format(
     xlim=(0, 210),
     ylim=(10,250),
     xticks=[], yticks=[],
     yticklabels=[], xticklabels=[],
     xtickminor=False, ygridminor=False,abc=True
)
fig.savefig("C:\\Users\\kiril\\Thesis\\CoBra\\cobra\\figs\\augmentations\\augmentations_000_strong.png",
    dpi=300, bbox_inches='tight')
#%%
r1 = load_arr('000450', which='c')
r2 = load_arr('000379', which='c')
r3 = load_arr('025202', which='c')
r4 = load_arr('005306', which='c')
r5 = load_arr('000402', which='c')
r6 = load_arr('006319', which='c')
d1 = load_arr('020131', which='c')
d2 = load_arr('024161', which='c')
#%%
fig, axs = pplt.subplots(ncols=3, nrows=1, share=False,
    wspace=('1pt','1pt',), abc=False,figsize=(9,7))
pplt.rc['abc.size']=18
# Augmented
axs[0].imshow(r1[50:-50,190,:], cmap='gray',aspect=.6)
axs[1].imshow(r2[50:-50,160,:], cmap='gray',aspect=.6)
axs[2].imshow(r3[30:-30,260,:], cmap='gray',aspect=.6)
#axs[3].imshow(r4[:-40,140,:], cmap='gray',aspect=.6)
#axs[4].imshow(d1[:-10,260,:], cmap='gray',aspect=.2)
#axs[5].imshow(d2[60:-60,10:-100,50].T, cmap='gray',aspect=1.2)
axs.format(
    xticks=[],
    yticks=[],
    yticklabels=[],
    xticklabels=[],
    xtickminor=False, ygridminor=False,
)
fig.savefig("C:\\Users\\kiril\\Thesis\\CoBra\\cobra\\figs\\augmentations\\ringing_deform_examples.png",
    dpi=300, bbox_inches='tight')