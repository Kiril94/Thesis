#%%
import shutil
import os
from os.path import join
from pathlib import Path
import pandas as pd
import numpy as np

#%%
disk_dir = "F:"
script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parent
data_dir = join(base_dir, 'data')
table_dir = join(data_dir, 'tables')
pat_groups_dir = join(data_dir, 'patient_groups')

dfc = pd.read_csv(join(table_dir, "neg_pos_clean.csv"))
cases_ls = np.loadtxt(join(pat_groups_dir, 
                't1_pre_post.txt'), dtype=str)
#%%
df3dt1 = dfc[(dfc.MRAcquisitionType=='3D') & \
    (dfc.Sequence=='t1') & (dfc.NumberOfSlices>=64)]
df_cases = df3dt1[df3dt1.PatientID.isin(cases_ls)]
df_pos_no_cases = df3dt1[~df3dt1.PatientID.isin(cases_ls)]

# %%
source_target_dirs = [
    (join(), target) for 
    ]