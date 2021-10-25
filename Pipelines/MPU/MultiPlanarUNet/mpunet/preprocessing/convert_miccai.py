#%%
# In[Import]
import nibabel as nib
nib.Nifti1Header.quaternion_threshold = -1e-06
import numpy as np
from bs4 import BeautifulSoup
sys.path.append("D:/Thesis/Pipelines")
from utilities import preprocess
import os 
from os.path import join, split
from pathlib import Path
script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parent.parent.parent.parent.parent.parent
Cobra_dir = join(base_dir, 'Cobra')
import sys
if Cobra_dir not in sys.path:
    sys.path.append(Cobra_dir)
from cobra.utilities import basic
import ast
import matplotlib.pyplot as plt
#%% 
# In[Paths]
data_path = join(base_dir,'Pipelines', 'data', 
    'mpunet', 'MPUnet_raw_data' ,'MICCAI')
images_tr = basic.list_subdir(join(data_path, 'train', 'images'))
images_ts = basic.list_subdir(join(data_path, 'test', 'images'))
images_val = basic.list_subdir(join(data_path, 'val', 'images'))
labels_tr = basic.list_subdir(join(data_path, 'train', 'labels'))
labels_ts = basic.list_subdir(join(data_path, 'test', 'labels'))
labels_val = basic.list_subdir(join(data_path, 'val', 'labels'))
paths_all = [images_tr, images_ts, images_val, labels_tr, labels_ts, labels_val]
print(os.listdir(data_path))

#%% 
# In[transform]
convert = False
def convert_image(im_path):
    im = nib.load(im_path)
    if len(im.shape)==3:
        new_im = np.expand_dims(im.get_fdata().astype(np.float32),-1)
        nib.save(nib.Nifti1Image(new_im, affine=im.affine), im_path)
        print(f"{im_path} converted")

for paths in paths_all:
    for im_path in paths:
        if convert:
            convert_image(im_path)
#%%
# In[Rename]
rename=False
label_files = [labels_tr, labels_ts, labels_val]
for files in label_files:
    for file in files:
        dir_, name = split(file)
        #name = "MICCAI_"+name[1:5]+'.nii'
        print(name)
        name = name[:6] + name[-4:]
        print(name)
        if rename:
            target_file = join(
                dir_, name)
            os.rename(file, target_file)
#%%
# In[Get the dict]
xml_dir = join(data_path, "labels_dict.xml")
with open(xml_dir, 'r') as f:
    data = f.read()
 
Bs_data = BeautifulSoup(data, "xml")
result_list = []
b_label = Bs_data.find_all('Label')
for b in b_label:
    number = int(b.find('Number').string)
    name = b.find('Name').string
    color = b.find('RGBColor').string
    result_list.append((number, name, color))
result_list = sorted(result_list)
result_list.insert(0, (0, 'background', '0 0 0'))
print(result_list[0])
print(b_label[0].find('Name').string)
with open(join(data_path, 'original_labels.txt'), 'w') as f:
    for item in result_list:
        f.write(f"{item[0]}, {item[1]}, {item[2]}\n")
orig_labels_dic = {tuple_[0]:tuple_[1] for tuple_ in result_list}
with open(f"{data_path}/original_labels_dic.txt", 'w') as f:
    print(orig_labels_dic, file=f)
#%%
# In[Convert labels to consecutive]
# The strategy is to fill missing labels by the last existing label,
# e.g. if 1 and 5 is missing we take 250 and make it 1, 249 is made 5
# List of relevant labels, all the rest is set to 0 (background)
labels_list = [46,4,49,51,50,52,23,36,57,55,
59,61,76,30,37,58,56,60,62,75,31,47,116,122,170,132,154,200,180,184,206,202,100,
138,166,102,172,104,136,146,178,112,118,120,124,140,152,186,142,162,164,190,204,
150,182,192,106,174,194,198,148,176,168,108,114,134,160,128,144,156,196,32,48,
117,123,171,133,155,201,181,185,207,203,101,139,167,103,173,105,137,147,179,
113,119,121,125,141,153,187,143,163,165,191,205,151,183,193,107,175,195,199,
149,177,169,109,115,135,161,129,145,157,197,44,45,11,35,38,40,39,41,71,72,73]
labels_list = sorted(labels_list)
trafo_dic = {x:(i+1) for i, x in enumerate(labels_list)}
# Remove all the irrelevant keys
irr_labels = list(set(orig_labels_dic.keys() - set(labels_list)))
new_labels_dic = orig_labels_dic.copy()
for key in irr_labels:
    new_labels_dic.pop(key)

new_labels_dic = preprocess.rename_keys(new_labels_dic, trafo_dic)
new_labels_dic[0] = 'background'
new_labels_dic = dict(sorted(
    new_labels_dic.items(),key=lambda x:x[0]))
with open(f"{data_path}/new_labels_dic.txt", 'w') as f:
    print(new_labels_dic, file=f)

#%%
# In[Get trafo dict for images]
for label in irr_labels: #map irrelevant labels to 0
    trafo_dic[label] = 0
print(trafo_dic)

with open(f"{data_path}/trafo_dic.txt", 'w') as f:
    print(trafo_dic, file=f)
#inv_trafo_dic = {v: k for k, v in trafo_dic.items()}
#%%
# In[Transform images]
labels_paths = labels_tr + labels_val + labels_ts
transform = False
if transform:
    for im_path in labels_paths:
        preprocess.transform_labels(im_path, trafo_dic)
