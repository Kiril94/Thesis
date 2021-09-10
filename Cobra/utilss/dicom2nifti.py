# -*- coding: utf-8 -*-

"""
  Module:  dicom_to_nifti.py
  01 Sep. 2021
  @author: Kiril
"""

import os
from pathlib import Path

script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parent.parent
dcm2nii_exe_path = os.path.join(base_dir, "helper\\dcm2niix_win\\dcm2niix.exe")


def dcm2nii(dcm_path, out_path, compression=6,op_sys=0):
    """Given dicom path and output path, converts dicom files in 1 folder
    to a nii file + json file containing the header. 
    The files are named corresponding
    to the folder name which is the SOPInstanceUID."""
    if (op_sys==0): #WINDOWS
            os.system(f"cmd /k {dcm2nii_exe_path} -{compression}\
              -f %f_%z -w 0 -o {out_path} {dcm_path}")
    elif (op_sys==1):  #LINUX  
        os.system(f'dcm2niix -w 0 -{compression} -f %f -o {out_path} {dcm_path}')
    else: 
        raise Exception('Non valid op_sys. Available options are 0 for Windows or 1 for Linux')



