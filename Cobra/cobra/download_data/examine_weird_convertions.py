#%%
import pandas as pd 
from os.path import join
from pathlib import Path
import os 

#%%
script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parents[1]  #cobra directory
table_dir = join(base_dir, 'tables')
filenames_list_path = "C:/Users/neus/Desktop/check_included_scans.txt"
name_corr_path = "F:/CoBra/Data/swi_nii/cmb_study/included_nii_v3_names.csv"
filenames_list2_path = "C:/Users/neus/Desktop/check_included_scans_v2.txt"

filenames_df = pd.read_csv(filenames_list_path)
names_corr_df = pd.read_csv(name_corr_path)
df_scan_info = pd.read_csv(join(table_dir,'swi_all.csv'))

filenames_df['filename'] = filenames_df['filename'].map(lambda x: str(x).zfill(6)+'.nii.gz')

merged_df = filenames_df.merge(names_corr_df,left_on='filename',right_on='new_name',how='inner')
merged_df = merged_df.merge(df_scan_info,on='SeriesInstanceUID',how='inner')

merged_df['PatientID'].to_csv(filenames_list2_path,index=False)

