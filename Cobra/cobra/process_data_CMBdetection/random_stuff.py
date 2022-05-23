# %%
from numpy import NaN
import pandas as pd 
from pathlib import Path
import nibabel as nib
import numpy as np
import os 
# %%
tables_path = Path(__file__).parent.parent / "tables" / "SynthCMB"
file_path_sytnh = tables_path / "sCMBInformationInfo.csv"
file_path_real = tables_path / "rCMBInformationInfo.csv"

df_synth = pd.read_csv(file_path_sytnh)
df_real = pd.read_csv(file_path_real)


df_synth['real NIFTI File Name'] = df_synth['NIFTI File Name'].str.split("_rsCMB").map(lambda x: x[0]+'.nii.gz')

#real CMB in sCMB scans
df_merged = df_synth.merge(df_real,how='inner',left_on='real NIFTI File Name',right_on='NIFTI File Name',validate='many_to_many',suffixes=('_synth','_real'))

#real CMB in sCMB slices
df_merged_same_slice = df_merged[ df_merged['z_position_synth']==df_merged['z_position_real']]

new_df = pd.DataFrame({'NIFTI File Name': df_merged_same_slice['NIFTI File Name_synth'],
                        'x_position': df_merged_same_slice['x_position_real'],
                        'y_position': df_merged_same_slice['y_position_real'],
                        'z_position': df_merged_same_slice['z_position_real'],
                        'real': True,
                        'synth_version': NaN})

new_df = new_df.drop_duplicates()
new_df.to_csv( tables_path / "rCMB_insCMB_scans_Info.csv", index=False)

all_scmb_data = pd.concat([df_synth.drop('real NIFTI File Name',axis='columns'),new_df])
all_scmb_data.to_csv(tables_path / "sCMBInformationInfo_plusrCMB.csv", index=False) 

# %%
tables_path = Path(__file__).parent.parent / "tables" / "SynthCMB"
info_path = tables_path / "rCMB_insCMB_scans_Info.csv"
all_info_path = tables_path / "all_info_splitted.csv"
all_info_path2 = tables_path / "all_info_splitted_v2.csv"

df_info = pd.read_csv(info_path)
# train_slices = len([ 1 for _,_ in df_train_new.groupby(['NIFTI File Name','z_position'])])
# test_slices = len([ 1 for _,_ in df_test_new.groupby(['NIFTI File Name','z_position'])])
# val_slices = len([ 1 for _,_ in df_val_new.groupby(['NIFTI File Name','z_position'])])

# print("Number of 2D-slices")
# print(f"train: {train_slices}\t test:{test_slices} \t val:{val_slices}")

# print("Number of 3D-scans")
# print(f"train: {df_train_new['NIFTI File Name'].unique().shape[0]}\t test:{df_test_new['NIFTI File Name'].unique().shape[0]} \t val:{df_val_new['NIFTI File Name'].unique().shape[0]}")

# print("Number of subjects")
# print(f"train: {df_train_new['SubjectID'].unique().shape[0]}\t test:{df_test_new['SubjectID'].unique().shape[0]} \t val:{df_val_new['SubjectID'].unique().shape[0]}")
df_all_info = pd.read_csv(all_info_path)
df_merged = df_info.merge(df_all_info,how='left',left_on=['NIFTI File Name','z_position'], right_on=['NIFTI File Name','z_position'], suffixes=('_new','_old'))

#take only rCMB slices
df_rCMB = df_merged.drop_duplicates(['NIFTI File Name','z_position','x_position_new','y_position_new','set','SubjectID'])
#add rCMB to all info splitted
new_to_all_info = pd.DataFrame({'NIFTI File Name': df_rCMB['NIFTI File Name'],
                                'x_position': df_rCMB['x_position_new'],
                                'y_position': df_rCMB['y_position_new'],
                                'z_position': df_rCMB['z_position'],
                                'real': True,
                                'synth_version': NaN ,
                                'SubjectID': df_rCMB['SubjectID'],
                                'set':df_rCMB['set']
                                })
all_info = pd.concat([df_all_info,new_to_all_info])
all_info.to_csv(all_info_path2,index=False)

#%%
tables_path = Path(__file__).parent.parent / "tables" / "SynthCMB"
all_info_path2 = tables_path / "all_info_splitted_v2.csv"

all_info = pd.read_csv(all_info_path2)

train_mask = (all_info['set']=='train')
test_mask = (all_info['set']=='test')
val_mask = (all_info['set']=='val')

print("Number of CMB")
print(f"train: {all_info[train_mask].shape[0]}\t test:{all_info[test_mask].shape[0]} \t val:{all_info[val_mask].shape[0]}")
print("Number of real CMB")
print(f"train: {all_info[train_mask&(all_info['real'])].shape[0]}\t test:{all_info[test_mask&(all_info['real'])].shape[0]} \t val:{all_info[val_mask&(all_info['real'])].shape[0]}")
print("Number of synth CMB")
print(f"train: {all_info[train_mask&(~all_info['real'])].shape[0]}\t test:{all_info[test_mask&(~all_info['real'])].shape[0]} \t val:{all_info[val_mask&(~all_info['real'])].shape[0]}")


train_slices = len([ 1 for _,_ in all_info[train_mask].groupby(['NIFTI File Name','z_position'])])
test_slices = len([ 1 for _,_ in all_info[test_mask].groupby(['NIFTI File Name','z_position'])])
val_slices = len([ 1 for _,_ in all_info[val_mask].groupby(['NIFTI File Name','z_position'])])

print("Number of 2D-slices")
print(f"train: {train_slices}\t test:{test_slices} \t val:{val_slices}")

print("Number of 3D-scans")
print(f"train: {all_info[train_mask]['NIFTI File Name'].unique().shape[0]}\t test:{all_info[test_mask]['NIFTI File Name'].unique().shape[0]} \t val:{all_info[val_mask]['NIFTI File Name'].unique().shape[0]}")

print("Number of subjects")
print(f"train: {all_info[train_mask]['SubjectID'].unique().shape[0]}\t test:{all_info[test_mask]['SubjectID'].unique().shape[0]} \t val:{all_info[val_mask]['SubjectID'].unique().shape[0]}")

# %%

tables_path = Path(__file__).parent.parent / "tables" / "SynthCMB"
info_noCMB = tables_path / "sCMBLocationInformationInfoNocmb.csv"
info_rCMB = tables_path / "rCMBInformationInfo.csv"
info_sCMB = tables_path / "sCMBInformationInfo_plusrCMB.csv"
info_all = tables_path / "all_info_splitted_v2.csv"

info1 = pd.read_csv(info_noCMB)
info2 = pd.read_csv(info_rCMB)
info3 = pd.read_csv(info_sCMB)
info_all = pd.read_csv(info_all)

new = pd.concat((info1,info2,info3))

subjectsID = info_all[info_all['real']]['NIFTI File Name'].str.split('_').map(lambda x: x[0])
print(len(subjectsID.unique()))

subjectsID = new['NIFTI File Name'].str.split('_').map(lambda x: x[0])
print(len(subjectsID.unique()))

print()

#%% 
# check what is in all_info_splitted_v2.csv
tables_path = Path(__file__).parent.parent / "tables" / "SynthCMB"
allinfo_path = tables_path / "all_info_splitted_v2.csv"
df = pd.read_csv(allinfo_path)

#%%
# Inspect volume metadata

def load_nifti_img(filepath, dtype=None):
    '''
    NIFTI Image Loader
    :param filepath: path to the input NIFTI image
    :param dtype: dataio type of the nifti numpy array
    :return: return numpy array
    '''
    nim = nib.load(filepath)
    out_nii_array = np.array(nim.get_data(),dtype=dtype)
    out_nii_array = np.squeeze(out_nii_array) # drop singleton dim in case temporal dim exists
    meta = {'affine': nim.get_affine(),
            'dim': nim.header['dim'],
            'pixdim': nim.header['pixdim'],
            'name': os.path.basename(filepath)
            }

    return out_nii_array, meta

path_v0 = "/home/neus/Documents/09.UCPH/MasterThesis/DATA/test_pipelines_cph_data/volumes/H000.nii.gz"
path_pred_v0 = "/home/neus/Documents/09.UCPH/MasterThesis/DATA/test_pipelines_cph_data/reshaped_probs_aug/H000.nii.gz"

v0,v0_meta = load_nifti_img(path_v0)

#%% 
# check slicing nocmb
tables_path = Path(__file__).parent.parent / "tables" / "SynthCMB"
slices_1ch = pd.read_csv(tables_path/"images.txt",header=None)
slices_3ch = pd.read_csv(tables_path/"images_3channel.txt",header=None)

slices_1ch['z'] = slices_1ch[0].str.split('_slice').map(lambda x: x[-1][:-7])
slices_3ch['z'] = slices_3ch[0].str.split('_slice').map(lambda x: x[-1][:-7])

notcommon = slices_1ch[ ~slices_1ch[0].isin(slices_3ch[0])]
common = slices_1ch[ slices_1ch[0].isin(slices_3ch[0])]

#%%
#save volume and mask with affine I
nib.openers.HAVE_INDEXED_GZIP=False
def load_nifti_img(filepath, dtype=None):
    '''
    NIFTI Image Loader
    :param filepath: path to the input NIFTI image
    :param dtype: dataio type of the nifti numpy array
    :return: return numpy array
    '''
    nim = nib.load(filepath)
    out_nii_array = np.array(nim.get_fdata(),dtype=dtype)
    out_nii_array = np.squeeze(out_nii_array) # drop singleton dim in case temporal dim exists
    meta = {'affine': nim.affine,
            'dim': nim.header['dim'],
            'pixdim': nim.header['pixdim'],
            'name': os.path.basename(filepath)
            }

    return out_nii_array, meta

#%%
cph_data_path = "/home/neus/Documents/09.UCPH/MasterThesis/DATA/test_pipelines_cph_data"
path_v0 = f"{cph_data_path}/volumes/H002.nii.gz"
path_pred_v0 = f"{cph_data_path}/reshaped_probs_attempt45/H002.nii.gz"


orig_vol,_ = load_nifti_img(path_v0)
mask,_ = load_nifti_img(path_pred_v0)

nib.save(nib.Nifti1Image(orig_vol,np.eye(4)),f"{cph_data_path}/H002_noAffine.nii.gz")
nib.save(nib.Nifti1Image(mask,np.eye(4)),f"{cph_data_path}/H002_mask_noAffine.nii.gz")

#%% count cmb from new data
import scipy.io as sio
new_folder_gt_path = "/media/neus/USB DISK/cmb-3dcnn-data/ground_truth"
files_in_gt_path = next(os.walk(new_folder_gt_path))[2]

new_folder_nii_path = "/media/neus/USB DISK/cmb-3dcnn-data/nii"
files_in_nii_path = next(os.walk(new_folder_nii_path))[2]

gt_num = []
shapes = []
names = []
pixdims = []
for idx in range(len(files_in_gt_path)):
    nii_name = files_in_nii_path[idx]
    gt_name = files_in_gt_path[idx]

    names.append(nii_name)

    img,meta = load_nifti_img(f"{new_folder_nii_path}/{nii_name}")
    print(img.shape)
    shapes.append([img.shape[0],img.shape[1],img.shape[2]])

    pixdims.append(meta['pixdim'][1:4])

    mat = sio.loadmat(f"{new_folder_gt_path}/{gt_name}")
    print(mat['gt_num'])
    gt_num.append(mat['gt_num'])

#%%
main_folder = "/media/neus/USB DISK/cmb-3dcnn-data"

gt_num = np.array(gt_num).squeeze()
shapes = np.array(shapes)
pixdims = np.array(pixdims)

df = pd.DataFrame({ 'file_name':names,
                    'n_cmb': gt_num,
                    'n_rows': shapes[:,0],
                    'n_cols': shapes[:,1],
                    'n_slices': shapes[:,2],
                    'x_res': pixdims[:,0],
                    'y_res': pixdims[:,1],
                    'z_res': pixdims[:,2],
                    })
df.to_csv(f"{main_folder}/volumes_info.csv",index=False)

#%%
names = []
positions = []
for idx in range(len(files_in_gt_path)):
    #load ground truth
    gt_name = files_in_gt_path[idx]
    mat = sio.loadmat(f"{new_folder_gt_path}/{gt_name}")

    #create row for each cmb
    for i_cmb in range(mat['gt_num'][0,0]):
        positions.append(mat['cen'][i_cmb])
        names.append(gt_name)

main_folder = "/media/neus/USB DISK/cmb-3dcnn-data"
positions = np.array(positions)
positions = positions -1 

df = pd.DataFrame({ 'file_name':names,
                    'x_pos': positions[:,0],
                    'y_pos': positions[:,1],
                    'z_pos': positions[:,2],
                    })
df.to_csv(f"{main_folder}/cmb_info.csv",index=False)

#%%
path_v0 = f"/media/neus/USB DISK/cmb-3dcnn-data/nii/17.nii"
path_pred_v0 =  f"/media/neus/USB DISK/cmb-3dcnn-data/masks/17.nii"


orig_vol,_ = load_nifti_img(path_v0)
mask,_ = load_nifti_img(path_pred_v0)

nib.save(nib.Nifti1Image(orig_vol,np.eye(4)),f"/media/neus/USB DISK/cmb-3dcnn-data/17_noAffine.nii.gz")
nib.save(nib.Nifti1Image(mask,np.eye(4)),f"/media/neus/USB DISK/cmb-3dcnn-data/17_mask_noAffine.nii.gz")

#%%
#%%
path_v0 = f"/home/neus/Documents/09.UCPH/MasterThesis/github/Thesis/Cobra/cobra/process_data_CMBdetection/prova2.nii.gz"
path_pred_v0 =  f"/home/neus/Documents/09.UCPH/MasterThesis/github/Thesis/Cobra/cobra/process_data_CMBdetection/prova2_mask.nii.gz"


orig_vol,_ = load_nifti_img(path_v0)
mask,_ = load_nifti_img(path_pred_v0)

nib.save(nib.Nifti1Image(orig_vol,np.eye(4)),f"/media/neus/USB DISK/prova2_noAffine.nii.gz")
nib.save(nib.Nifti1Image(mask,np.eye(4)),f"/media/neus/USB DISK/prova2_mask_noAffine.nii.gz")

#%%
