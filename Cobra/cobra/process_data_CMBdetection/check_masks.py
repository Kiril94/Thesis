
import numpy as np
import nibabel as nib
import os 
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


cmb_slices_path = "/home/lkw918/cobra/data/Synth_CMB_sliced_new"

sets = ['train','test','val']
views = ['axial','sagittal','coronal']

log_file = open(f"{cmb_slices_path}/wrong_n_channels.csv",'w')
log_file.write("filename,path,shape_x,shape_y,shape_z\n")

for view_name in views:

    for set_name in sets:

        path = f"{cmb_slices_path}/{set_name}/{view_name}/masks/"
        files = next(os.walk(path))[2]

        for file in files:
            file_path = f"{path}/{file}"
            img,_ = load_nifti_img(file_path)

            if (len(img.shape)==3):
                log_file.write(f"{file},{file_path},{img.shape[0]},{img.shape[1]},{img.shape[2]}\n")
