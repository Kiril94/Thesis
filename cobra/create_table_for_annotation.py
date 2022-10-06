#%%
import pandas as pd
import pickle as pkl
import json
import os
#%%
df = pd.read_csv("G:\\CoBra\\Data\\metadata\\tables\\neg_pos_clean_nos.csv",
            usecols=['PatientID', 'SeriesInstanceUID'])
with open("G:\\CoBra\\Data\\metadata\\tables\\disk_series_directories.json", 'rb') as f:
    dir_dic = json.load(f)
#%%
with open("G:\\CoBra\\Data\\metadata\\tables\\newIDs_dic.pkl", 'rb') as f:
    newIDs_dic = pkl.load(f)
segmented_path = "C:\\Users\\guest_acc\\CoBra\\data\\prediction-new"
segmented_ids = [f[:6] for f in os.listdir(segmented_path) if f.endswith("1mm.nii.gz")]
newIDs_dic = {k: v for k, v in newIDs_dic.items() if v in segmented_ids}
#%%
df['new_sid'] = df.SeriesInstanceUID.map(newIDs_dic)
df = df.dropna(subset=['new_sid'])
#%%

new_pid_dic = {old_pid:str(new_pid).zfill(6) for new_pid, old_pid in enumerate(df.PatientID.unique())}
df['new_pid'] = df.PatientID.map(new_pid_dic)
df = df.reindex(columns = ['new_pid','new_sid','PatientID','SeriesInstanceUID'])
df = df.sort_values(['new_pid', 'new_sid'], ascending=[True, False])
df.to_csv("C:\\Users\\guest_acc\\CoBra\\data\\longitudinal_to_label\\pid_sid.csv", index=None)