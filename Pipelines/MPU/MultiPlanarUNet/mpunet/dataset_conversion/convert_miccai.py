#%%
# In[Import]
import nibabel as nib
nib.Nifti1Header.quaternion_threshold = -1e-06
import numpy as np
import os 
from os.path import join, split
from pathlib import Path
script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parent.parent.parent.parent.parent.parent
Cobra_dir = join(base_dir, 'Cobra')
import sys
if Cobra_dir not in sys.path:
    sys.path.append(Cobra_dir)
from cobra.utilities import basic

#%% 
# In[Paths]
data_path = join(base_dir,'Pipelines', 'data', 
    'mpunet', 'MPUnet_raw_data' ,'MICCAI')
images_tr = basic.list_subdir(join(data_path, 'train', 'images'))
images_ts = basic.list_subdir(join(data_path, 'test', 'images'))
images_val = basic.list_subdir(join(data_path, 'val', 'images'))
labels_tr = basic.list_subdir(join(data_path, 'train', 'labels'))
labels_ts = basic.list_subdir(join(data_path, 'test', 'labels'))
labels_val = basic.list_subdir(join(data_path, 'val', 'labels'))
paths_all = [images_tr, images_ts, images_val, labels_tr, labels_ts, labels_val]
print(os.listdir(data_path))

#%% 
# In[transform]
def convert_image(im_path):
    im = nib.load(im_path)
    if len(im.shape)==3:
        new_im = np.expand_dims(im.get_fdata().astype(np.float32),-1)
        nib.save(nib.Nifti1Image(new_im, affine=im.affine), im_path)
        print(f"{im_path} converted")

for paths in paths_all:
    for im_path in paths:
        convert_image(im_path)
#%%
# In[Rename]
rename=False
label_files = [labels_tr, labels_ts, labels_val]
for files in label_files:
    for file in files:
        dir_, name = split(file)
        #name = "MICCAI_"+name[1:5]+'.nii'
        print(name)
        name = name[:6] + name[-4:]
        print(name)
        if rename:
            target_file = join(
                dir_, name)
            os.rename(file, target_file)