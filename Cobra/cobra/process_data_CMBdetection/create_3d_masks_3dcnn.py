"""
created on march 26th 2022
author: Neus Rodeja Ferrer
"""

#%%
import pandas as pd 
import os 
import nibabel as nib
import numpy as np 
import time
import datetime

# FOR SYNTH CMB 
#cmb_info_path = "/home/neus/Documents/09.UCPH/MasterThesis/github/Thesis/Cobra/cobra/tables/SynthCMB/all_info_splitted_v2.csv"
cmb_info_path = "/home/lkw918/cobra/data/cmb-3dcnn-data/cmb_info.csv"
cmb_nii_path = "/home/lkw918/cobra/data/cmb-3dcnn-data/nii"
cmb_masks_path = "/home/lkw918/cobra/data/cmb-3dcnn-data/masks"

# #configuration for SYNTH
# r_xy_patch, r_z_patch = 6,3
# border_xy_patch, border_z_patch = 2,1
# frac_cmb_thresh = 0.52

#configuration for 3dCNN
r_xy_patch, r_z_patch = 12,6
border_xy_patch, border_z_patch = 4,2
frac_cmb_thresh = 0.65

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
    meta = {'affine': nim.affine, #nim.get_affine(),
            'dim': nim.header['dim'],
            'pixdim': nim.header['pixdim'],
            'name': os.path.basename(filepath)
            }

    return out_nii_array, meta 

#%% #create masks

#read cmb info
df_cmb_info = pd.read_csv(cmb_info_path)

#CONFIGURATION FOR 3d CNN
df_cmb_info['path'] = df_cmb_info['file_name'].map(lambda x: f"{cmb_nii_path}/{x[:-4].zfill(2)}.nii") #specific for this case


#create thresh function
apply_thresh = np.vectorize(lambda x: 1 if x>frac_cmb_thresh else 0)


#find volumes
vol_groups = df_cmb_info.groupby("file_name")

start_time = time.time()
for idx_vol,vol_info in vol_groups:
    filename = f"{vol_info['file_name'].values[0][:-4].zfill(2)}.nii"
    print(filename)
    #read nifti
    #path = f"{cmb_nii_path}/{vol_info['group_folder'].values[0]}/{filename}" #FOR SYNTH CMB
    path = f"{cmb_nii_path}/{filename}" #FOR 3dCNN
    img,meta = load_nifti_img(path)
    mask = np.zeros_like(img)

    for idx_cmb,cmb_info in vol_info.iterrows():
        
        x,y,z = cmb_info['x_pos'],cmb_info['y_pos'],cmb_info['z_pos']
        intensity_cmb = img[x,y,z]
        
        #define patch around image
        slice_x = slice(x-r_xy_patch,x+r_xy_patch+1)
        slice_y = slice(y-r_xy_patch,y+r_xy_patch+1)
        slice_z = slice(z-r_z_patch,z+r_z_patch+1)
        patch_around_cmb = img[slice_x,slice_y,slice_z]

        #take outer border of the patch (8 cares del cub)
        mid_xy,mid_z = int(r_xy_patch/2),int(r_z_patch/2)
        slice_x1 = patch_around_cmb[:border_xy_patch,:,:].flatten()
        slice_x2 = patch_around_cmb[-border_xy_patch:,:,:].flatten()
        slice_y1 = patch_around_cmb[:,:border_xy_patch,:].flatten()
        slice_y2 = patch_around_cmb[:,-border_xy_patch:,:].flatten()
        slice_z1 = patch_around_cmb[:,:,:border_z_patch].flatten()
        slice_z2 = patch_around_cmb[:,:,-border_z_patch:].flatten()
        intensity_rest = np.mean( np.concatenate( (slice_x1,slice_x2,slice_y1,slice_y2,slice_z1,slice_z2)) )

        #create mask on x,y,z
        mask[x,y,z]= 1
        frac_cmb = (intensity_rest-patch_around_cmb)/(intensity_rest-intensity_cmb)
        
        # print(frac_cmb.shape)
        # print(img.shape)
        # print(x,y,z)
        proposed_mask = apply_thresh(frac_cmb)

        #eliminate artifacts 
        centered_x_axis = proposed_mask[:,r_xy_patch,r_z_patch]
        centered_y_axis = proposed_mask[r_xy_patch,:,r_z_patch]
        centered_z_axis = proposed_mask[r_xy_patch,r_xy_patch,:]

        x_axis_left = np.argwhere( centered_x_axis[:r_xy_patch]==0 )
        x_axis_right = np.argwhere( centered_x_axis[r_xy_patch:]==0 )
        x_min = np.max(x_axis_left) if len(x_axis_left)>0 else None
        x_max = np.min(x_axis_right) + r_xy_patch if len(x_axis_right)>0 else None

        y_axis_left = np.argwhere( centered_y_axis[:r_xy_patch]==0 )
        y_axis_right = np.argwhere( centered_y_axis[r_xy_patch:]==0 )
        y_min = np.max(y_axis_left) if len(y_axis_left)>0 else None
        y_max = np.min(y_axis_right) + r_xy_patch if len(y_axis_right)>0 else None

        z_axis_left = np.argwhere( centered_z_axis[:r_z_patch]==0 )
        z_axis_right = np.argwhere( centered_z_axis[r_z_patch:]==0 )
        z_min = np.max(z_axis_left) if len(z_axis_left)>0 else None
        z_max = np.min(z_axis_right) + r_z_patch if len(z_axis_right)>0 else None

        #set to 0 everything outside the limits
        proposed_mask[slice(None,None if x_min is None else x_min +1 ),:,:] = 0
        proposed_mask[slice(x_max,None),:,:] = 0

        proposed_mask[:,slice(None,None if y_min is None else y_min +1 ),:] = 0
        proposed_mask[:,slice(y_max,None),:] = 0

        proposed_mask[:,:,slice(None,None if z_min is None else z_min +1 )] = 0
        proposed_mask[:,:,slice(z_max,None)] = 0

        mask[slice_x,slice_y,slice_z] = proposed_mask

    nib.save(nib.Nifti1Image(mask,meta['affine']),f"{cmb_masks_path}/{filename}")
    
    print(str(datetime.timedelta(seconds= (time.time()-start_time))))