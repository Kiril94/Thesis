# -*- coding: utf-8 -*-
"""
Created on Mon Nov  30 12:06:00 2021

@author: neusRodeja
"""
import sys
sys.path.insert(0, '/home/neus/Documents/09.UCPH/MasterThesis/github/Thesis/Cobra/cobra/')

from access_sif_data.load_data_tools import load_nifti_array_dim
from numpy import NaN
import pandas as pd
from glob import iglob 
main_path = "/home/neus/Documents/09.UCPH/MasterThesis/DATA/Synthetic Cerebral Microbleed on SWI images/PublicDataShare_2020"
rCMB_doc = "rCMBInformationInfo.xlsx"
sCMB_doc = "sCMBInformationInfo.xlsx"
sCMB_fromNoCMB_doc = "sCMBLocationInformationInfoNocmb.xlsx"

main_folder = "/home/neus/Documents/09.UCPH/MasterThesis/github/Thesis/Cobra/cobra"
new_tables_path = f"{main_folder}/tables/SynthCMB"

data_path = "/home/neus/Documents/09.UCPH/MasterThesis/DATA/Synthetic Cerebral Microbleed on SWI images/PublicDataShare_2020/sCMB_NoCMBSubject"

def convert_table(file_path,nCMB_max,synth=True):
    #Dictionary of equivalences for the column names
    key_dict = {'NIFTI File Name':'NIFTI File Name ',
                        'CMB_1_x':'Location CMB # 1',
                        'CMB_1_y':'Unnamed: 2',
                        'CMB_1_z':'Unnamed: 3',
                        'CMB_2_x':'Location CMB # 2',
                        'CMB_2_y':'Unnamed: 5',
                        'CMB_2_z':'Unnamed: 6',
                        'CMB_3_x':'Location CMB # 3',
                        'CMB_3_y':'Unnamed: 8',
                        'CMB_3_z':'Unnamed: 9',
                        'CMB_4_x':'Location CMB # 4',
                        'CMB_4_y':'Unnamed: 11',
                        'CMB_4_z':'Unnamed: 12',
                        'CMB_5_x':'Location CMB # 5',
                        'CMB_5_y':'Unnamed: 14',
                        'CMB_5_z':'Unnamed: 15',
                        'CMB_6_x':'Location CMB # 6',
                        'CMB_6_y':'Unnamed: 17',
                        'CMB_6_z':'Unnamed: 18',
                        'CMB_7_x':'Location CMB # 7',
                        'CMB_7_y':'Unnamed: 20',
                        'CMB_7_z':'Unnamed: 21',
                        'CMB_8_x':'Location CMB # 8',
                        'CMB_8_y':'Unnamed: 23',
                        'CMB_8_z':'Unnamed: 24',
                        'CMB_9_x':'Location CMB # 9',
                        'CMB_9_y':'Unnamed: 26',
                        'CMB_9_z':'Unnamed: 27',
                        'CMB_10_x':'Location CMB # 10',
                        'CMB_10_y':'Unnamed: 29',
                        'CMB_10_z':'Unnamed: 30',
                        'CMB_11_x':'Location CMB # 11',
                        'CMB_11_y':'Unnamed: 32',
                        'CMB_11_z':'Unnamed: 33',
                        'CMB_12_x':'Location CMB # 12',
                        'CMB_12_y':'Unnamed: 35',
                        'CMB_12_z':'Unnamed: 36',
                        'CMB_13_x':'Location CMB # 13',
                        'CMB_13_y':'Unnamed: 38',
                        'CMB_13_z':'Unnamed: 39',
                        'CMB_14_x':'Location CMB # 14',
                        'CMB_14_y':'Unnamed: 41',
                        'CMB_14_z':'Unnamed: 42',
                        'CMB_15_x':'Location CMB # 15',
                        'CMB_15_y':'Unnamed: 44',
                        'CMB_15_z':'Unnamed: 45',
                        'CMB_16_x':'Location CMB # 16',
                        'CMB_16_y':'Unnamed: 47',
                        'CMB_16_z':'Unnamed: 48',
                        'CMB_17_x':'Location CMB # 17',
                        'CMB_17_y':'Unnamed: 50',
                        'CMB_17_z':'Unnamed: 51',
                        'CMB_18_x':'Location CMB # 18',
                        'CMB_18_y':'Unnamed: 53',
                        'CMB_18_z':'Unnamed: 54',
                        'CMB_19_x':'Location CMB # 19',
                        'CMB_19_y':'Unnamed: 55',
                        'CMB_19_z':'Unnamed: 56',
                        'LocationCirclewholeReal_58':'LocationCirclewholeReal_58',
                        'LocationCirclewholeReal_59':'LocationCirclewholeReal_59',
                        'LocationCirclewholeReal_60':'LocationCirclewholeReal_60',
                        }

    r_cmb_df = pd.read_excel(file_path)
    #Create new dataframe for CMB information (1row, 1CMB)
    column_names = ['NIFTI File Name','SubjectID','T_ID','x_position','y_position','z_position','real','synth_version']
    new_df = pd.DataFrame(columns=column_names)
    for i in range(nCMB_max,0,-1):
        mask = (r_cmb_df[key_dict[f'CMB_{i}_x']].astype(str).str.isspace())
        cmbs_data = r_cmb_df[~mask]

        if cmbs_data.shape[0]>0:

            if (synth):
                new_rows = pd.DataFrame( {'NIFTI File Name': cmbs_data[key_dict['NIFTI File Name']],
                                'SubjectID':[int(x.split('_')[0]) for x in cmbs_data[key_dict['NIFTI File Name']]],
                                'T_ID':[int(x.split('_')[1][1]) for x in cmbs_data[key_dict['NIFTI File Name']]],
                                'x_position':cmbs_data[key_dict[f'CMB_{i}_x']].astype(int),
                                'y_position':cmbs_data[key_dict[f'CMB_{i}_y']].astype(int),
                                'z_position':cmbs_data[key_dict[f'CMB_{i}_z']].astype(int),
                                'real':False,
                                'synth_version':[x.split('_')[-1][:-7] for x in cmbs_data[key_dict['NIFTI File Name']]]
                }
                )            
            else:
                new_rows = pd.DataFrame( {'NIFTI File Name': cmbs_data[key_dict['NIFTI File Name']],
                                'SubjectID':[int(x.split('_')[0]) for x in cmbs_data[key_dict['NIFTI File Name']]],
                                'T_ID':[int(x.split('_')[1][1]) for x in cmbs_data[key_dict['NIFTI File Name']]],
                                'x_position':cmbs_data[key_dict[f'CMB_{i}_x']].astype(int),
                                'y_position':cmbs_data[key_dict[f'CMB_{i}_y']].astype(int),
                                'z_position':cmbs_data[key_dict[f'CMB_{i}_z']].astype(int),
                                'real':True,
                                'synth_version':NaN
                }
                )
            new_df = pd.concat([new_df,new_rows],ignore_index=True)

    return new_df

def get_dimensions(data_folder):
    nifti_paths_sCMB_noCMB = [file for file in iglob(f"{data_folder}/*.nii.gz")]

    column_names = ['NIFTI File Name','Width','Height','Depth']
    meta_data = pd.DataFrame(columns=column_names)
    for file in nifti_paths_sCMB_noCMB:

        x_dim,y_dim,z_dim = load_nifti_array_dim(file,dtype=None)

        meta = pd.DataFrame({'NIFTI File Name':[file.split('/')[-1]],
                            'Width':[x_dim],
                            'Height':[y_dim],
                            'Depth':[z_dim],
                            })

        meta_data = pd.concat([meta_data,meta])
    return meta_data 


#Convert rCMB
new_df = convert_table(f"{main_path}/{rCMB_doc}",19,synth=False)
new_df.to_csv(f'{new_tables_path}/{rCMB_doc[:-5]}.csv',index=False)

# #Convert sCMB
# new_df = convert_table(f"{main_path}/{sCMB_doc}",10)
# new_df.to_csv(f'{new_tables_path}/{sCMB_doc[:-5]}.csv',index=False)

# #Conver sCMB noCMB
# new_df = convert_table(f"{main_path}/{sCMB_fromNoCMB_doc}",10)
# new_df.to_csv(f'{new_tables_path}/{sCMB_fromNoCMB_doc[:-5]}.csv',index=False)

# # Get dimensions sCMB no CMB
# meta_data = get_dimensions(data_path)
# meta_data.to_csv(f"{new_tables_path}/nifti_metadata.csv")


