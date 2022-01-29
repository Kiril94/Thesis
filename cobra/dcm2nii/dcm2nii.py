# -*- coding: utf-8 -*-

"""
  Module:  dicom_to_nifti.py
  01 Sep. 2021
  @author: Kiril
"""

import os
from pathlib import Path
import subprocess as sp

script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parent.parent
dcm2nii_exe_path = os.path.join(
    base_dir, "dcm2nii\\dcm2niix_win\\dcm2niix.exe")
def convert_dcm2nii(dcm_path, out_path, compression=3, verbose=0, op_sys=0,
            output_filename='%j', gz_compress='y'):
    """Given dicom path and output path, converts dicom files in 1 folder
    to a nii file + json file containing the header. sp.call waits until conversion is finished.
    op_sys: 0 for Windows, 1 for Linux'
    Compression level 3 gives the best time-size tradeoff
    The file name is constructed as follows:
        PatientId_ProtocolName_SequenceName(0018,1020)_SequenceName(0018,0024) 
        ProtocolName:
            User-defined description of the conditions 
            under which the Series was performed.
        SequenceName (0018,0024):
            so a T1 scan converted with "myName%z" might yield "myNameT1"
    Main arguments:
    -f : filename (%a=antenna (coil) name, %b=basename, %c=comments, 
        %d=description, %e=echo number, %f=folder name, %g=accession number, 
        %i=ID of patient, %j=seriesInstanceUID, %k=studyInstanceUID, 
        %m=manufacturer, %n=name of patient, %o=mediaObjectInstanceUID, 
        %p=protocol, %r=instance number, %s=series number, %t=time, 
        %u=acquisition number, %v=vendor, %x=study ID; %z=sequence name; 
        default '%f_%p_%t_%s')
    
    Other arguments:
        -a: adjacent DICOMs (images from same series always in same folder) 
            for faster conversion (n/y, default n)
        -l: losslessly scale 16-bit integers to use dynamic range (y/n/o 
            [yes=scale, no=no, but uint16->int16, o=original], default n)
        -z: gz compress images (y/i/n/3, default n)
        -i: ignore derived, localizer and 2D images (y/n, default n)
        --progress: report progress (y/n, default n)
        """
    
    if (op_sys == 0):  # WINDOWS
        command = f"cmd /k {dcm2nii_exe_path} -{compression} -a y\
              -f {output_filename} -l y -v {verbose} -o {out_path} {dcm_path}"
        sp.call([dcm2nii_exe_path, '-a','y', '-i', 'y',
              '-f', output_filename, '-l','y', '-v', str(verbose), '-o', out_path, dcm_path], timeout=600)
    elif (op_sys == 1):  # LINUX
        os.system(
            f'dcm2niix -w 0 -{compression} -a y -l y -v {verbose} -z y -f -f %f_%p_%z -o {out_path} {dcm_path}')
    else:
        raise Exception(
            'Non valid op_sys. Available options are 0 for Windows or 1 for Linux')

