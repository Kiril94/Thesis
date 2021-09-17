"""
@author: klein
"""
import numpy as np
from pydicom import dcmread
import os
import vis
from glob import iglob
from pathlib import Path
import datetime as dt
#from utilss.basic import DotDict



BASE_DIR = Path(__file__).resolve().parent.parent
data_dir = f"{BASE_DIR}/data"

patient_dirs = os.listdir(BASE_DIR)
patient_dir = f"{data_dir}/patient1"

def get_docs_path_list(scan_dir):
    reports = iglob(f"{scan_dir}/*/*/*/DOC/*/*.pdf")
    reports_list = [x for x in reports] 
    return reports_list

def reconstruct3d(scan_dir):
    """
    Combines the pixel data contained in dicom files of a scan to a 3d array.
    Returns: array of shape (num_slices, h, w)
    """
    files = os.listdir(scan_dir)
    dicom = [dcmread(f"{scan_dir}/{f}") for f in files]  # all slices
    try:
        slice_loc = [float(d.SliceLocation) for d in dicom]
        indices_sort = np.argsort(np.array(slice_loc))
        arr2d = np.array([d.pixel_array for d in dicom])
        arr3d = arr2d[indices_sort]
    except:
        print("Slicelocation could not be found\
              for at least one of the dicom files,\
                  returning empty arr")
        arr3d = np.empty((1,1))
    return arr3d


def get_patient_key_list():
    return [('PatientID', 'str'), ('PatientSex','str'), ('Positive', 'boolean')]

def get_scan_key_list():
    """"List containing keys and corresponding datatype as tuples."""
    key_list = [('SeriesInstanceUID','str'),
                ('StudyInstanceUID','str'),
                ('PatientID', 'str'), 
                ('AngioFlag', 'str'),
                ('AcquisitionMatrix','str'),
                ('AcquisitionContrast', 'str'),
                ('AcquisitionDuration', 'float'),
                ('AcquisitionMatrix','list'),
                ('dBdt','float'),
                ('EchoTime','float'),
                ('EchoTrainLength', 'int'),
                ('EchoNumbers', 'int'),
                ('FlipAngle', 'float'),
                ('FrameOfReferenceUID', 'str'),
                ('ImagingFrequency', 'float'),
                ('ImagedNuclues', 'str'),
                ('InstanceCreationDate','date'), 
                ('InstanceCreationTime','time'),
                ('InversionTime','float'),
                ('ImagesInAcquisition', 'int'),
                ('ImageType','str'),
                ('MagneticFieldStrength', 'float'),
                ('Manufacturer', 'str'), 
                ('ManufacturerModelName', 'str'),
                ('MRAcquisitionType', 'str'),
                ('NumberofAverages','float'),
                ('NumberOfEchoes', 'int'),
                ('NumberofPhaseEncodingSteps', 'int'),
                ('PatientPosition','str'),
                ('PixelSpacing','list'),
                ('PixelBandwith','float'),
                ('PixelPresentation', 'str'),
                ('PixelSpacing', 'str'),
                ('PhotometricInterpretation','str'),
                ('PulseSequenceName', 'str'),
                ('RepetitionTime', 'float'),
                ('Rows', 'int'),
                ('Columns', 'int'),
                ('ScanningSequence', 'str'), 
                ('SequenceVariant','str'),
                ('SequenceName', 'str'),
                ('ScanOptions','str'), 
                ('SeriesDescription', 'str'),
                ('SoftwareVersions','str'),
                ('SliceThickness', 'float'),
                ('StudyPriorityID', 'str'), 
                ('PatientPosition', 'str'),
                ('SpacingBetweenSlices','float'),
                ('SecondEcho', 'float'),
                ('VariableFlipAngleFlag', 'str')
                ]
    return key_list

def get_scan_dictionary(scan_dir, reconstruct_3d=True):
    """Returns a dictionary for scan at scan_dir"""
    if len(os.listdir(scan_dir))!=0:
        try:
            dicom_file_dir = os.path.join(scan_dir, os.listdir(scan_dir)[0])
            dicom = dcmread(dicom_file_dir)
        except:
            print('Dicom file non readable')
            dicom = None
            for file_num in range(len(os.listdir(scan_dir))):
                try:
                    dicom_file_dir = os.path.join(scan_dir, os.listdir(scan_dir)[file_num])
                    dicom = dcmread(dicom_file_dir)
                    break
                except:
                    print('Dicom file non readable')
                    continue
    else:
        dicom = None
        print(f"{scan_dir} is empty")
        
    key_list = get_scan_key_list()
    
    if reconstruct_3d:
        scan_dict = {'arr3d': reconstruct3d(scan_dir)}
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


class Patient():

    def __init__(self, patient_dir):
        self.patient_dir = patient_dir
        self.patient_id = os.path.split(patient_dir)[1] 
        self.patient_id = os.path.split(patient_dir)[1]
    
    def get_id(self):
        return self.patient_id
    

    def info(self):
        """Returns dictionary with general info about the patient."""
        subdir = os.path.join(self.patient_dir, os.listdir(self.patient_dir)[0])
        mr_dir = os.path.join(subdir, "MR")
        mr_subdir = os.path.join(mr_dir, os.listdir(mr_dir)[0])
        
        dicom = dcmread(os.path.join(mr_subdir, os.listdir(mr_subdir)[0]))
        
        patient_sex = getattr(dicom, "PatientSex")
        if patient_sex=='':
            patient_sex=None
            
        if (subdir.split('/')[-3]=='positive'):
            positive = True
        else: 
            positive = False
            
        info_dict = {'PatientID': self.patient_id,
                     'PatientSex': patient_sex,
                     'Positive': positive, }
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

    def get_scan_dictionary(self, scan_number, reconstruct_3d=True):
        """Returns a dictionary for scan with scan_number, if reconstruct """
        scan_directories = self.get_scan_directories()
        scan_dir = scan_directories[scan_number]
        if len(os.listdir(scan_dir))!=0:
            try:
                dicom_file_dir = os.path.join(scan_dir, os.listdir(scan_dir)[0])
                dicom = dcmread(dicom_file_dir)
            except:
                print('Dicom file non readable')
                dicom = None
                for file_num in range(len(os.listdir(scan_dir))):
                    try:
                        dicom_file_dir = os.path.join(scan_dir, os.listdir(scan_dir)[file_num])
                        dicom = dcmread(dicom_file_dir)
                        break
                    except:
                        print('Dicom file non readable')
                        continue
        else:
            dicom = None
            print(f"{scan_dir} is empty")
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
        all_scan_dicts = [self.get_scan_dictionary(scan_number, reconstruct_3d=reconstruct_3d)\
                          for scan_number in range(len(scan_directories))]
        return all_scan_dicts
    
    def show_scan(self, scan_number, args):
        """Shows slices of 3d volume. For args look at vis.display3d"""
        scan_directories = self.get_scan_directories()
        scan_dir = scan_directories[scan_number]
        arr3d = self.reconstruct3d(scan_dir)
        vis.display3d(arr3d, **args)
            
            
            