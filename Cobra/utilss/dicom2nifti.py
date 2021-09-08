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

def dcm2nii(dcm_path, out_path, compression=8, verbose=0):
    """Given dicom path and output path, converts dicom files in 1 folder
    to a nii file + json file containing the header. 
    The file name is constructed as follows:
        PatientId_ProtocolName_SequenceName(0018,1020)_SequenceName(0018,0024) 
        ProtocolName:
            User-defined description of the conditions 
            under which the Series was performed.
        SequenceName (0018,1020):
            For example, the output filename "myName%q" would convert a 
            Spin Echo sequence to be "myNameSE.nii" 
            (new feature, in versions from 30Aug2015)
        SequenceName (0018,0024):
            so a T1 scan converted with "myName%z" might yield "myNameT1"
        """
    os.system(f"cmd /k {dcm2nii_exe_path} -{compression} -a y\
              -f %f_%p_%q_%z -l y -v {verbose} -z y -o {out_path} {dcm_path}")
