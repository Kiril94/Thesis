"""Check which brain regions were not segmented correctly"""

#%% 
import pandas as pd
import shutil
from os.path import join

#%%
df = pd.read_csv("G:\\CoBra\\Data\\volume_pred_results\\volume_prediction_results_new.csv")

#%%
df0 = df[df.isin([0]).any(axis=1)]
#df0.to_csv("G:\\CoBra\\Data\\volume_pred_results\\volume_prediction_results_zeros.csv", index=False)
df_inspect = df0.sample(200)
#%%
ids = [str(nid).zfill(6) for nid in df_inspect.newID]
segmented_paths = ["G:\\CoBra\\Data\\volume_cross_nii\\prediction-new", 
                   "G:\\CoBra\\Data\\volume_longitudinal_nii\\prediction-new"]
tgt_dir = "F:\\failed_segment\\zeros"
for id in ids:
    try:
        shutil.copy(join(segmented_paths[0], id+"_1mm_seg.nii.gz"),tgt_dir)
        shutil.copy(join(segmented_paths[0], id+"_1mm.nii.gz"),tgt_dir)
    except:
        try:
            shutil.copy(join(segmented_paths[1], id+"_1mm_seg.nii.gz"),tgt_dir)
            shutil.copy(join(segmented_paths[1], id+"_1mm.nii.gz"),tgt_dir)
        except:
            print("Not found: ", id)
