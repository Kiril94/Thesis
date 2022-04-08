#%%
import os, sys
from os.path import split, join
base_dir = split(os.getcwd())[0]
if base_dir not in sys.path:
    sys.path.append(base_dir)
from pathlib import Path
import nibabel as nib
from utilities import basic
nib.Nifti1Header.quaternion_threshold = -1e-06
#%%
thesis_dir = Path(base_dir).parent
aug_dir = join(thesis_dir, 'MRI-Augmentation')
data_dir = join(base_dir, 'data')
#%%
import matlab.engine
eng = matlab.engine.start_matlab()
eng.addpath(join(aug_dir, '3D'), nargout=0)
eng.addpath(join(aug_dir, '3D', 'bias_field'))
eng.addpath(join(aug_dir, '3D', 'elastic_deform'))
eng.addpath(join(aug_dir, '3D', 'gibbs_ringing'))
eng.addpath(join(aug_dir, '3D', 'motion_ghosting'))
#%%
# In[Load test image]
img_dir = join(data_dir, "augmentations\\MICCAI\\imagesTr")
lbl_dir = join(data_dir,"augmentations\\MICCAI\\labelsTr")
imga_dir = join(data_dir, "augmentations\\MICCAI\\imagesTr_aug")
lbla_dir = join(data_dir,"augmentations\\MICCAI\\labelsTr_aug")
tst_img = nib.load(basic.list_subdir(img_dir)[0])
tst_lbl = nib.load(basic.list_subdir(lbl_dir)[0])

#%%
# In[Lets try for one img]
X = tst_img.get_fdata()
aff = tst_img.affine
#%%
# In[Try augmentation]
def normalize(X):
    xmin, xmax = X.min(), X.max()
    return (X-xmin)/(xmax-xmin), xmin, xmax
def renormalize(X, xmin, xmax):
    return X*(xmax-xmin)+xmin
Xn, xmin, xmax = normalize(X)
X1 = eng.bias_field(Xn)

#%%
Xorg = renormalize(Xn, xmin,xmax)
tst_img = nib.Nifti1Image(Xorg, aff)
# eng.bias_field()
nib.save(tst_img, join(imga_dir, 'test.nii.gz'))

#%%
print(1-0.2**4)