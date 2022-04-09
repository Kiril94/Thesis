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
# import matlab.engine
# eng = matlab.engine.start_matlab()
# eng.addpath(join(aug_dir, '3D'), nargout=0)
# eng.addpath(join(aug_dir, '3D', 'bias_field'))
# eng.addpath(join(aug_dir, '3D', 'elastic_deform'))
# eng.addpath(join(aug_dir, '3D', 'gibbs_ringing'))
# eng.addpath(join(aug_dir, '3D', 'motion_ghosting'))
#%%
# In[Load test image]
img_dir = join(data_dir, "augmentations\\MICCAI\\imagesTr")
lbl_dir = join(data_dir,"augmentations\\MICCAI\\labelsTr")
imga_dir = join(data_dir, "augmentations\\MICCAI\\imagesTr_aug")
lbla_dir = join(data_dir,"augmentations\\MICCAI\\labelsTr_aug")
tst_img = nib.load(basic.list_subdir(img_dir)[0])
tst_lbl = nib.load(basic.list_subdir(lbl_dir)[0])
tst_img_a = nib.load(basic.list_subdir(imga_dir)[0])
tst_lbl_a = nib.load(basic.list_subdir(lbla_dir)[0])

#%%
# In[Rename and add header to images]
for tr_img_path in basic.list_subdir(img_dir):
    tr_img = split(tr_img_path)[1]
    num = tr_img[8:10]
    new_num = str(int(num) + 50)
    new_tr_img = tr_img[:8] + new_num + tr_img[10:]
    
    X_aug = nib.load(join(imga_dir, tr_img))
    X_org = nib.load(tr_img_path)
    img_with_hdr = nib.Nifti1Image(X_aug.get_fdata(), X_org.affine)
    print(tr_img)
    print(new_tr_img)
    nib.save(img_with_hdr, join(imga_dir, new_tr_img))
    os.remove(join(imga_dir, tr_img))


#% 
# In[]
# In[Rename and add header to labels]
for tr_lbl_path in basic.list_subdir(lbl_dir):
    tr_lbl = split(tr_lbl_path)[1]
    num = tr_lbl[8:10]
    new_num = str(int(num) + 50)
    new_tr_lbl = tr_lbl[:8] + new_num + tr_lbl[10:]
    
    Y_aug = nib.load(join(lbla_dir, tr_lbl))
    Y_org = nib.load(tr_lbl_path)
    lbl_with_hdr = nib.Nifti1Image(Y_aug.get_fdata(), Y_org.affine)
    print(tr_lbl)
    print(new_tr_lbl)
    nib.save(lbl_with_hdr, join(lbla_dir, new_tr_lbl))
    os.remove(join(lbla_dir, tr_lbl))
# Xa = tst_img_a.get_fdata()
# aff = tst_img.affine

# tst_img = nib.Nifti1Image(Xa, aff)
# eng.bias_field()
# nib.save(tst_img, join(imga_dir, 'test2.nii.gz'))


# tst_img = nib.Nifti1Image(Xorg, aff)
# eng.bias_field()
# nib.save(tst_img, join(imga_dir, 'test.nii.gz'))

# %%
