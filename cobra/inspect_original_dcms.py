#%%
import pickle
import pandas as pd
import json
#df = pd.read_csv('output_list.txt', sep=" ", header=None, names=["a", "b", "c"])
#PosixPath('Second/Third/Fourth/Fifth')
with open("C:\\Users\\kiril\\Thesis\\CoBra\\cobra\\data\\tables\\newIDs_dic.pkl", 'rb') as f:
    newIDs_dic = pickle.load(f)
with open("C:\\Users\\kiril\\Thesis\\CoBra\\cobra\\data\\tables\\disk_series_directories_new.json") as f:
  sdirs_dic = json.load(f)
#df = pd.read_csv()
keys = [k for k, v in newIDs_dic.items() if v == '193153']
print(keys)
sid = keys[0]
print(sdirs_dic[sid])

#%%

df = pd.read_csv("C:\\Users\\kiril\\Thesis\\CoBra\\cobra\\data\\tables\\scan_tables\\scan_after_sq_pred_dst_nos_date.csv")
#%%
df[df.newID==193153].keys()
df[df.newID==193153].iloc[0]