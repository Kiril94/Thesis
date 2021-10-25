import nibabel as nib
nib.Nifti1Header.quaternion_threshold = -1e-06
import numpy as np


def rename_keys(d, trafo_dic):
    """Keys of d are renamed according to trafo_dic,
    a copy is returned"""
    d_new = d.copy()
    for item in trafo_dic.items():
        d_new[item[1]] = d_new.pop(item[0])
    return d_new

def transform_labels(im_path, trafo_dic):
    """Label nii is transorfmed according to trafo_dic"""
    im = nib.load(im_path)
    arr = im.get_fdata().astype(np.int32)
    new_arr = np.vectorize(trafo_dic.get)(arr)
    nib.save(nib.Nifti1Image(new_arr, affine=im.affine), im_path)