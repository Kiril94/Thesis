"""
created on march 26th 2022
author: Neus Rodeja Ferrer
"""

#%%
import pandas as pd 
import os 
import nibabel as nib
import numpy as np 

cmb_info_path = "/home/neus/Documents/09.UCPH/MasterThesis/github/Thesis/Cobra/cobra/tables/SynthCMB/all_info_splitted_v2.csv"
# cmb_info_path = "/home/lkw918/cobra/data/Synthetic_Cerebral_Microbleed_on_SWI_images/all_info_splitted_v2.csv"
cmb_nii_path = "/home/lkw918/cobra/data/Synthetic_Cerebral_Microbleed_on_SWI_images/PublicDataShare_2020" # "/media/neus/USB DISK/cmb-3dcnn-data/nii"
cmb_masks_path = "/home/lkw918/cobra/data/Synthetic_Cerebral_Microbleed_on_SWI_images/PublicDataShare_2020/3d_masks"
cmb_slices_path = "/home/lkw918/cobra/data/Synth_CMB_sliced_new"

#configuration
r_xy_patch, r_z_patch = 13,3
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
#df_cmb_info['path'] = df_cmb_info['file_name'].map(lambda x: f"{cmb_nii_path}/{x[:-4].zfill(2)}.nii") #specific for this case

dict_group_folders = {'sCMB':'sCMB_NoCMBSubject',
                    'rCMB':'rCMB_DefiniteSubject',
                    'rsCMB':'sCMB_DefiniteSubject',
                    }
df_cmb_info['group'] = df_cmb_info['NIFTI File Name'].str.split('_').map(lambda x: x[7] if len(x)==9 else 'rCMB' )
df_cmb_info['group_folder'] = df_cmb_info['group'].map(lambda x: dict_group_folders[x])
#df_cmb_info['path'] = f"{cmb_nii_path}/{df_cmb_info['group_folder']}/{df_cmb_info['NIFTI File Name']}"
df_cmb_info.rename(columns = {'NIFTI File Name': 'file_name'},inplace=True)

#create thresh function
apply_thresh = np.vectorize(lambda x: 1 if x>frac_cmb_thresh else 0)

#find volumes
vol_groups = df_cmb_info.groupby("file_name")

for idx_vol,vol_info in vol_groups:
    filename = vol_info['file_name'].values[0]
    print(filename)
    #read nifti
    path = f"{cmb_nii_path}/{vol_info['group_folder'].values[0]}/{filename}"
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
    
    if (vol_info['set'].values[0] == 'test'): #save all slices

        #save sagittal slices
        for x in range(img.shape[0]):
            slice_img = img[:,:,x]
            slice_msk = mask[:,:,x]

            slice_name = f'{filename[:-7]}_slice{x}.nii.gz'
            path_to_save = f"{cmb_slices_path}/{vol_info['set'].values[0]}/sagittal"
            nib.save(nib.Nifti1Image(slice_img,np.eye(4)),f"{path_to_save}/images/{slice_name}")
            nib.save(nib.Nifti1Image(slice_msk,np.eye(4)),f"{path_to_save}/masks/{slice_name}")

        #save coronal slices
        for y in range(img.shape[1]):
            slice_img = img[:,:,y]
            slice_msk = mask[:,:,y]

            slice_name = f'{filename[:-7]}_slice{y}.nii.gz'
            path_to_save = f"{cmb_slices_path}/{vol_info['set'].values[0]}/coronal"
            nib.save(nib.Nifti1Image(slice_img,np.eye(4)),f"{path_to_save}/images/{slice_name}")
            nib.save(nib.Nifti1Image(slice_msk,np.eye(4)),f"{path_to_save}/masks/{slice_name}")

        #save axial slices
        for z in range(img.shape[2]):
            slice_img = img[:,:,z]
            slice_msk = mask[:,:,z]

            slice_name = f'{filename[:-7]}_slice{z}.nii.gz'
            path_to_save = f"{cmb_slices_path}/{vol_info['set'].values[0]}/axial"
            nib.save(nib.Nifti1Image(slice_img,np.eye(4)),f"{path_to_save}/images/{slice_name}")
            nib.save(nib.Nifti1Image(slice_msk,np.eye(4)),f"{path_to_save}/masks/{slice_name}")

    else: #save only slices with positives

        #save sagittal slices
        for x in range(img.shape[0]):
            slice_msk = mask[:,:,x]

            if (np.any(slice_msk)==1):
                slice_img = img[:,:,x]
                
                slice_name = f'{filename[:-7]}_slice{x}.nii.gz'
                path_to_save = f"{cmb_slices_path}/{vol_info['set'].values[0]}/sagittal"
                nib.save(nib.Nifti1Image(slice_img,np.eye(4)),f"{path_to_save}/images/{slice_name}")
                nib.save(nib.Nifti1Image(slice_msk,np.eye(4)),f"{path_to_save}/masks/{slice_name}")

        #save coronal slices
        for y in range(img.shape[1]):
            slice_msk = mask[:,:,y]

            if (np.any(slice_msk)==1):
                slice_img = img[:,:,y]

                slice_name = f'{filename[:-7]}_slice{y}.nii.gz'
                path_to_save = f"{cmb_slices_path}/{vol_info['set'].values[0]}/coronal"
                nib.save(nib.Nifti1Image(slice_img,np.eye(4)),f"{path_to_save}/images/{slice_name}")
                nib.save(nib.Nifti1Image(slice_msk,np.eye(4)),f"{path_to_save}/masks/{slice_name}")

        #save axial slices
        for z in range(img.shape[2]):
            slice_msk = mask[:,:,z]

            if (np.any(slice_msk)==1):
                slice_img = img[:,:,z]
                
                slice_name = f'{filename[:-7]}_slice{z}.nii.gz'
                path_to_save = f"{cmb_slices_path}/{vol_info['set'].values[0]}/axial"
                nib.save(nib.Nifti1Image(slice_img,np.eye(4)),f"{path_to_save}/images/{slice_name}")
                nib.save(nib.Nifti1Image(slice_msk,np.eye(4)),f"{path_to_save}/masks/{slice_name}")
