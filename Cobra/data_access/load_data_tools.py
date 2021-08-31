import numpy as np
from pydicom import dcmread
import os
from vis import vis
from glob import iglob
from pathlib import Path
import datetime as dt
from utils.utils import dotdict



BASE_DIR = Path(__file__).resolve().parent.parent
data_dir = f"{BASE_DIR}/data"

patient_dirs = os.listdir(BASE_DIR)
patient_dir = f"{data_dir}/patient1"



def reconstruct3d(scan_dir):
    """
    Combines the pixel data contained in dicom files of a scan to a 3d array.
    Returns: array of shape (num_slices, h, w)
    """
    files = os.listdir(scan_dir)
    dicom = [dcmread(f"{scan_dir}/{f}") for f in files]  # all slices
    slice_loc = [float(d.SliceLocation) for d in dicom]
    indices_sort = np.argsort(np.array(slice_loc))
    arr2d = np.array([d.pixel_array for d in dicom])
    arr3d = arr2d[indices_sort]
    return arr3d

def get_patient_key_list():
    return [('PatientID', 'str'), ('PatientSex','str')]

def get_scan_key_list():
    """"List containing keys and corresponding datatype as tuples."""
    key_list = [('InstanceCreationDate','date'), 
                ('InstanceCreationTime','time'),
                ('Manufacturer', 'str'), ('ManufacturerModelName', 'str'),
                ('ScanningSequence', 'str'), ('SequenceVariant','str'),
                ('ScanOptions','str'), ('MRAcquisitionType', 'str'),
                ('AngioFlag', 'str'), 
                ('SliceThickness', 'float'),
                ('RepititionTime', 'float'),
                ('EchoTime','float'),
                ('NumberofAverages','float'),
                ('ImagingFrequency', 'float'),
                ('EchoNumbers', 'int'),
                ('MagneticFieldStrength', 'float'),
                ('NumberofPhaseEncodingSteps', 'int'),
                ('EchoTrainLength', 'int'),
                ('FlipAngle', 'float'),
                ('SpacingBetweenSlices','float'),
                ('ImagesInAcquisition', 'int') ]
    return key_list


class Patient():

    def __init__(self, patient_dir):
        self.patient_dir = patient_dir
        self.patient_id = os.path.split(patient_dir)[1]
        
    def info(self):
        """Returns dictionary with general info about the patient."""
        subdir = os.path.join(self.patient_dir, os.listdir(self.patient_dir)[0])
        mr_dir = os.path.join(subdir, "MR")
        mr_subdir = os.path.join(mr_dir, os.listdir(mr_dir)[0])
        dicom = dcmread(os.path.join(mr_subdir, os.listdir(mr_subdir)[0]))
        patient_sex = getattr(dicom, "PatientSex")
        if patient_sex=='':
            patient_sex=None
        info_dict = {'PatientID': self.patient_id,
                     'PatientSex': patient_sex }
        return info_dict
        
    def get_scan_directories(self):
        """Helper function which returns all MR subdirectories contained in a 
        patients folder."""
        subdirectories = iglob(f"{self.patient_dir}/*/MR/*")
        subdirectories_list = [x for x in subdirectories if os.path.isdir(x)]
        return subdirectories_list
    
    def reconstruct3d(self, scan_dir):
        """
        Combines the pixel data contained in dicom files of a scan subdir 
        to a 3d array.
        Returns: array of shape (num_slices, h, w)
        """
        files = os.listdir(scan_dir)
        dicom = [dcmread(f"{scan_dir}/{f}") for f in files]  # all slices
        slice_loc = [float(d.SliceLocation) for d in dicom]
        indices_sort = np.argsort(np.array(slice_loc))
        arr2d = np.array([d.pixel_array for d in dicom])
        arr3d = arr2d[indices_sort]
        return arr3d

    def scan_dictionary(self, scan_number, reconstruct_3d=True):
        """Returns a dictionary for scan with scan_number, if reconstruct """
        scan_directories = self.get_scan_directories()
        scan_dir = scan_directories[scan_number]
        dicom_file_dir = os.path.join(scan_dir, os.listdir(scan_dir)[0])
        dicom = dcmread(dicom_file_dir)
        key_list = get_scan_key_list()
        
        if reconstruct_3d:
            scan_dict = {'arr3d': self.reconstruct3d(scan_dir)}
        else:
            scan_dict = {}
        for k in key_list:
            try: # see if dicom contains this tag
                value = getattr(dicom, k[0])
                if value=='':
                    value = None
                elif k[1]=='str':
                    value = str(value)
                elif k[1]=='int':
                    value = int(value)
                elif k[1]=='float':
                    value = float(value)
                elif k[1]=='date':
                    value = dt.date(int(value[:4]), int(value[4:6]), int(value[6:]))
                elif k[1]=='time':
                    value = dt.time(int(value[:2]), int(value[2:4]), int(value[4:6]))
                else:
                    print(f"Unknown Datatype in get key list: {k[1]}")
            except:
                value = None

            scan_dict[k[0]] = value
        return dotdict(scan_dict)
    
    def all_scan_dicts(self, reconstruct_3d=True):
        """Returns list with all scan dictionaries for a patient."""
        scan_directories = self.get_scan_directories()
        all_scan_dicts = [self.scan_dictionary(scan_number, reconstruct_3d=reconstruct_3d)\
                          for scan_number in range(len(scan_directories))]
        return all_scan_dicts
    
    def show_scan(self, scan_number, args):
        """Shows slices of 3d volume. For args look at vis.display3d"""
        scan_directories = self.get_scan_directories()
        scan_dir = scan_directories[scan_number]
        arr3d = self.reconstruct3d(scan_dir)
        vis.display3d(arr3d, **args)
            
            
            