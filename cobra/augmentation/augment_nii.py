#%%
import os, sys
from os.path import split, join
base_dir = split(os.getcwd())[0]
if base_dir not in sys.path:
    sys.path.append(base_dir)
from pathlib import Path
import nibabel as nib
#%%
thesis_dir = Path(base_dir).parent
aug_dir = join(thesis_dir, 'MRI-Augmentation')
#%%
import matlab.engine
eng = matlab.engine.start_matlab()
eng.addpath(join(aug_dir, '3D', 'bias_field'), nargout=0)
eng.addpath(join(aug_dir, '3D', 'elastic_deform'), nargout=0)
eng.addpath(join(aug_dir, '3D', 'gibbs_ringing'), nargout=0)
eng.addpath(join(aug_dir, '3D', 'motion_ghosting'), nargout=0)
#%%
# In[Load test image]
img = nib.load("F:\\CoBra\\Data\\volume_longitudinal_nii\\input\\nii_files\\006725.nii.gz")
X = img.get_fdata()
#%%
# In[Try augmentation]
eng.bias_field()