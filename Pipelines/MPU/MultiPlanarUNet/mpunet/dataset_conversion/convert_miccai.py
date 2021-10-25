#%%
# In[Import]
import nibabel as nib
nib.Nifti1Header.quaternion_threshold = -1e-06
import numpy as np
from bs4 import BeautifulSoup
import gzip
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
with open(join(data_path, 'labels.txt'), 'w') as f:
    for item in result_list:
        f.write(f"{item[0]}, {item[1]}, {item[2]}\n")
labels_dic = {tuple_[0]:tuple_[1] for tuple_ in result_list}

#%%
# In[Convert labels to consecutive]
# The strategy is to fill missing labels by the last existing label,
# e.g. if 1 and 5 is missing we take 250 and make it 1, 249 is made 5
# List of labels will come, ignore all the others
def label_count(im_paths):
    count_dic = {}
    for label in range(208):
        counter = 0
        for im_path in im_paths:
            im = nib.load(im_path).get_fdata().astype(np.int32)
            counter += np.sum(im==label)
        print(f"{label}, {counter}")
        count_dic[label] = counter
    return count_dic
labels_tr_val = labels_tr + labels_val
# check which labels are not present in the data
count_dic = label_count(labels_tr_val)
#%%
trafo_dic = {label:label+1 for label in labels_dic.keys()}
inv_trafo_dic = {v: k for k, v in trafo_dic.items()}
#%%
def transform_labels(im_path, trafo_dic):
    im = nib.load(im_path).get_fdata().astype(np.int32)
    im = np.vectorize(trafo_dic.get)(im)
    return im
# max_label = np.max(np.array([x[0] for x in result_list]))
new_arr = transform_labels(labels_tr[0],trafo_dic)
print(new_arr)

print(np.vectorize(inv_trafo_dic.get)(new_arr))
#%%
print(np.max(np.array([x[0] for x in result_list])))
#print(len(result_list))
#print(len([0.7706, 0.8767, 0.0, 0.7676, 0.7688, 0.7213, 0.7761, 0.9253, 0.915, 0.9106, 0.9397, 0.9394, 0.9349, 0.9347, 0.0, 0.0, 0.9564, 0.9552, 0.8862, 0.8663, 0.8589, 0.6857, 0.6296, 0.9146, 0.9149, 0.878, 0.8705, 0.9283, 0.9236, 0.8849, 0.8858, 0.8476, 0.8523, 0.0, 0.0, 0.0, 0.8705, 0.8133, 0.8437, 0.5679, 0.5414, 0.8359, 0.7982, 0.8537, 0.8615, 0.4576, 0.4802, 0.6348, 0.5149, 0.7164, 0.714, 0.8003, 0.804, 0.645, 0.6542, 0.6435, 0.6912, 0.7607, 0.7762, 0.5419, 0.3745, 0.7921, 0.7269, 0.6925, 0.7699, 0.7387, 0.7019, 0.7969, 0.7524, 0.7523, 0.7169, 0.6171, 0.6616, 0.856, 0.8076, 0.459, 0.7436, 0.847, 0.8007, 0.7133, 0.5937, 0.7478, 0.765, 0.2182, 0.4254, 0.5975, 0.7168, 0.8052, 0.7892, 0.8271, 0.7669, 0.7072, 0.6531, 0.4641, 0.5039, 0.7296, 0.6237, 0.6547, 0.5528, 0.7182, 0.7967, 0.6842, 0.7367, 0.7782, 0.7516, 0.8473, 0.8236, 0.7206, 0.7236, 0.7544, 0.7449, 0.6879, 0.7736, 0.6531, 0.7523, 0.8246, 0.7656, 0.6662, 0.6858, 0.7705, 0.7293, 0.8053, 0.8279, 0.7946, 0.8236, 0.6417, 0.5896, 0.5852, 0.4651, 0.7431, 0.7384, 0.847, 0.7993, 0.8667, 0.8453, 0.6824, 0.619, 0.6059, 0.7126] ))
#%%
# In[]

#%%

a = np.array([[1,2,3],
              [3,2,4]])
my_dict = {1:23, 2:34, 3:36, 4:45}
#print(np.vectorize(my_dict.get)(a))
#print(os.listdir(data_path))       
