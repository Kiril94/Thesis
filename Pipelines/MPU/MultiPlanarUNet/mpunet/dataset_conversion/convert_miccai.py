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
xml_dir = join(task_folder, "labels_dict.xml")
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

print(result_list[0])
#print(dir(b_label[0].find('Name')))
print(b_label[0].find('Name').string)
with open(join(task_folder, 'labels.txt'), 'w') as f:
    for item in result_list:
        f.write(f"{item[0]}, {item[1]}, {item[2]}\n")


#%%
# In[Convert labels to consecutive]
a = np.array([[1,2,3],
              [3,2,4]])
my_dict = {1:23, 2:34, 3:36, 4:45}
print(np.vectorize(my_dict.get)(a))
print(os.listdir(data_path))       
def convert_to_consecutive(im):

    pass
