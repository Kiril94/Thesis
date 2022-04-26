#%%
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
#%%
t1_nii = nib.load("F:\\CoBra\\Data\\volume_cross_nii\\input\\nii_files\\segmented\\325594.nii.gz")
arr = t1_nii.get_fdata()
#%%
plt.style.use('dark_background')
fig, ax = plt.subplots(1,10)
ax = ax.flatten()
for i, a in enumerate(ax):
    a.imshow(arr[35:-20,140+2*i,16:-13], cmap='gray')
    a.axis('off')
    a.set_position((-i/12,i/20,1,1), which='both')
fig.savefig('t1_3d.png', dpi=900,  bbox_inches="tight")
#%%
plt.imshow(arr[180,:,:])