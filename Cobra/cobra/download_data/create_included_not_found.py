"""
"""
#%%
import pandas as pd
from pathlib import Path
from os.path import join
import os

#%%
list_seriesuid = ['9d4e01787bbfe1da2056efa2fbd5b3c5',
'203eb99be47a32c361481136896303c9',
'168091681c46aeb8d2dafc42a30e6d65',
'0569d2217d7ba26e9801f8bf78eb76e8',
'119c20c821c973ecc4abfdde00a41fdd',
'498f5653b3c7ca101edc39435fe20f2c',
'96948c17b964fe9177e5a8f081b8fd7d',
'c8a411565f74abc9ce257d732fa751a3',
'075784a75c16ebb42cabb6d54eeeed1b',
'ac11f79a7709a623309beba66656e537']

script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parents[1]  #cobra directory
table_dir = join(base_dir, 'tables')

swi_info  = pd.read_csv(join(table_dir,'swi_all.csv'))

swi_info = swi_info[ swi_info['SeriesInstanceUID'].isin(list_seriesuid)]

print([x for x in list_seriesuid if (not x in swi_info['SeriesInstanceUID'].tolist())])
swi_info['PatientID'].to_csv(join(table_dir,'swi_included_not_found.csv'),index=False)

#%%
#from repository
script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parents[1]  #cobra directory
table_dir = join(base_dir, 'tables')
input_file = join(table_dir,'ids_swi_excluded.csv')

download = join(table_dir,'log_downloaded_swi.txt')
to_download = join(table_dir,'log_to_download_swi.txt')

excl_df = pd.read_csv(input_file)
down_df = pd.read_csv(download)
to_down_df = pd.read_csv(to_download)
# %%
print(excl_df['PatientID'].shape, excl_df['PatientID'].unique().shape)
print(down_df['PatientID'].shape, down_df['PatientID'].unique().shape)
print(to_down_df['PatientID'].shape, to_down_df['PatientID'].unique().shape)


# %%
