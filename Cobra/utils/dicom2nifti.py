# -*- coding: utf-8 -*-

"""
  Module:  dicom_to_nifti.py
  01 Sep. 2021
  @author: Kiril
"""

import os


def dcm2nii(dcm_path, out_path):
    """Given dicom path and output path, converts dicom files in 1 folder
    to a nii file + json file containing the header. 
    The files are named corresponding
    to the folder name which is the SOPInstanceUID."""
    os.system(f"cmd /k D:\\Thesis\\Cobra\\helper\\dcm2niix_win\\dcm2niix.exe \
              -f %f -o {out_path} {dcm_path}")

