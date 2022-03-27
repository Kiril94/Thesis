"""
created on march 1st 2022
autor: neus rodeja ferrer
"""

import pandas as pd
from pathlib import Path
import random
import math

random.seed(42)

train_size,val_size,test_size = 0.7,0.1,0.2

input_table = Path(__file__).parent.parent / "tables" / "SynthCMB" / "all_info.csv"
output_table = Path(__file__).parent.parent / "tables" / "SynthCMB" / "all_info_splitted.csv"
df = pd.read_csv(input_table)


df['SubjectID'] =  df['NIFTI File Name'].str.split('_').map(lambda x: x[0]) 

groups = [ group_df.sort_values('NIFTI File Name') for _,group_df in df.groupby("SubjectID")]
random.shuffle(groups)

df = pd.concat(groups).reset_index(drop=True)

n_cmb = df.shape[0]

n_train = math.floor(n_cmb*train_size)
n_val = math.floor(n_cmb*val_size)
n_test = n_cmb - n_train - n_val

df_train = df[:n_train] #.reset_index(drop=True)
df_val = df[n_train:(n_train+n_val)] #.reset_index(drop=True)
df_test = df[(n_train+n_val):] #.reset_index(drop=True)

#ensure that the train/test/val do NOT have same subjects
last_subject_train = df_train.iloc[-1]['SubjectID']
last_subject_val = df_val.iloc[-1]['SubjectID']

from_val_to_train = df_val[ df_val['SubjectID']==last_subject_train]
from_test_to_val = df_test[ df_test['SubjectID']==last_subject_val]

df_train_new = pd.concat([df_train,from_val_to_train]).reset_index(drop=True)
df_train_new['set'] = 'train'
df_val_new = df_val.drop(index=from_val_to_train.index)
df_val_new = pd.concat([df_val_new,from_test_to_val]).reset_index(drop=True)
df_val_new['set'] = 'val'
df_test_new = df_test.drop(index=from_test_to_val.index)
df_test_new['set'] = 'test'

df_new = pd.concat([df_train_new,df_val_new,df_test_new])
df_new.to_csv(output_table,index=False)

print("Number of CMB")
print(f"train: {df_train_new.shape[0]}\t test:{df_test_new.shape[0]} \t val:{df_val_new.shape[0]}")

train_slices = len([ 1 for _,_ in df_train_new.groupby(['NIFTI File Name','z_position'])])
test_slices = len([ 1 for _,_ in df_test_new.groupby(['NIFTI File Name','z_position'])])
val_slices = len([ 1 for _,_ in df_val_new.groupby(['NIFTI File Name','z_position'])])

print("Number of 2D-slices")
print(f"train: {train_slices}\t test:{test_slices} \t val:{val_slices}")

print("Number of 3D-scans")
print(f"train: {df_train_new['NIFTI File Name'].unique().shape[0]}\t test:{df_test_new['NIFTI File Name'].unique().shape[0]} \t val:{df_val_new['NIFTI File Name'].unique().shape[0]}")

print("Number of subjects")
print(f"train: {df_train_new['SubjectID'].unique().shape[0]}\t test:{df_test_new['SubjectID'].unique().shape[0]} \t val:{df_val_new['SubjectID'].unique().shape[0]}")

