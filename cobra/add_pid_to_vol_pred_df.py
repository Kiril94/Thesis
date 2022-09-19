"""Add patient id to volume prediciton df"""
#%%
import pandas as pd
import gzip
from os.path import join
import pickle
#%%
table_dir = "D:\\Thesis\\CoBra\\cobra\\data\\tables"
# loading gzip table
with gzip.open(join(table_dir, 'scan_3dt1_clean.gz'), 'rb') as f:
    df = pickle.load(f)
df_vol = pd.read_csv("D:\\Thesis\\CoBra\\cobra\\data\\volume_prediction_results_new.csv")
df.head()
print(len(df_vol))
print(len(df))
df_vol = pd.merge(df[['PatientID', 'SeriesInstanceUID']], df_vol, 
    on='SeriesInstanceUID', how='right')
print(len(df_vol))
df_vol.head()
df_vol.to_csv("D:\\Thesis\\CoBra\\cobra\\data\\volume_prediction_results_new.csv", index=False)