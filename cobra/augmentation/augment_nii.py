#%%
import os, sys
from os.path import split, join
base_dir = split(os.getcwd())[0]
if base_dir not in sys.path:
    sys.path.append(base_dir)
from pathlib import Path
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
