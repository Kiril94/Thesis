#%%
import shutil
from os.path import join, split
#%%
results_file_path = "G:\\CoBra\\Data\\volume_cross_nii\\results.txt"
with open (results_file_path, "r") as myfile:
    data = myfile.read().splitlines()

for line in data[5:]:
    file = line[-14:-1]
    try:
        shutil.move(join("G:\\CoBra\\Data\\volume_cross_nii\\input\\nii_files\\segmented",  file),
            "G:\\CoBra\\Data\\volume_cross_nii\\input\\nii_files\\segmented\\failed")
    except:
        pass