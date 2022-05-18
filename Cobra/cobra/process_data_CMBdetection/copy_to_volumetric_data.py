"""
created on april 19th 2022
author: Neus Rodeja Ferrer

script to copy files from the original folder to our structured training folder
"""

from pathlib import Path
import pandas as pd 
import shutil 

data_path = Path("/home/lkw918/cobra/data")
sliced_folder = data_path / "Synth_CMB_sliced"
volumes_folder = data_path / "Synthetic_Cerebral_Microbleed_on_SWI_images"/"PublicDataShare_2020"
volumes_out_folder = data_path / "volumetric_data" / "original"
#volumes_folder = "/media/neus/USB DISK/Synthetic_Cerebral_Microbleed_on_SWI_images/PublicDataShare_2020"
all_info_path = sliced_folder / "all_info_splitted_v2.csv"
#all_info_path = Path(__file__).parent.parent /"tables"/"SynthCMB"/"all_info_splitted_v2.csv"

df_info = pd.read_csv(all_info_path)
df_info['group'] = df_info['NIFTI File Name'].str.split('_').map(lambda x: x[7] if len(x)==9 else 'rCMB' )

df_vols = df_info.groupby(['NIFTI File Name','z_position'])

dict_group_folders = {'sCMB':'sCMB_NoCMBSubject',
                    'rCMB':'rCMB_DefiniteSubject',
                    'rsCMB':'sCMB_DefiniteSubject',
                    }

for idx,vol in df_vols:

    file_name = vol['NIFTI File Name'].values[0]
    group_folder = dict_group_folders[vol['group'].values[0]]
    set = vol['set'].values[0]

    img_input_path = Path(volumes_folder) / group_folder / file_name
    img_output_path = Path(volumes_out_folder) / set / "images" / file_name

    msk_input_path = Path(volumes_folder) / "3d_masks" / file_name
    msk_output_path = Path(volumes_out_folder) / set / "masks" / file_name

    shutil.copy(img_input_path,img_output_path)
    shutil.copy(msk_input_path,msk_output_path)