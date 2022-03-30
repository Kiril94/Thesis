
import os 
from os.path import join
from pydicom import dcmread
from cobra.dcm2nii import dcm2nii
import time 
from cobra.download_data.basic import get_proc_id,remove_file

def check_dicoms(src_path, sif_src_path):
    """Checks whether all the dcm files from sif were downloaded."""
    if len(os.listdir(src_path))==len(os.listdir(sif_src_path)):
        return 0
    else: return 1
    
def check_if_philips(src_path):
    """Given the src_path of the dicoms, check if the Manufacturer is Philips"""
    dcm_dirs = [join(src_path, f) for f in os.listdir(src_path)]
    found = False
    n_missing = 0
    manufacturer = 'Unknown'
    while not found and n_missing<=len(dcm_dirs):
        try:
            manufacturer = get_value_from_header(dcm_dirs[n_missing], 'Manufacturer')
            found = True
        except:
            n_missing+=1
    if 'Philips' in manufacturer:
        return True
    else:
        return False
    
def get_value_from_header(dcm_dir, key):
    """Read value from dcm header stored under key"""
    dcm = dcmread(dcm_dir)
    return dcm[key].value

def dcm2nii_safe(disk_dcm_path, nii_out_path, sid, test,  timeout=2000):
    "Only keeps niis if dcm2nii converter returns 0"
    if test:
            print(disk_dcm_path + " exists, start nii conversion")
            start=time.time()
    print(get_proc_id(test), " Convert from disk")
    if (not os.path.exists(nii_out_path)): os.makedirs(nii_out_path)
    dcm2nii_out = dcm2nii.convert_dcm2nii(
        disk_dcm_path, nii_out_path, verbose=0, op_sys=0,
                output_filename='%j', create_info_json='n', gz_compress='y',
                timeout=timeout)
    if test:
            print("The conversion took "+str(round(time.time()-start,3))+'s')
            if dcm2nii_out==1:
                print("conversion from disk fail")
            else:
                print("conversion from disk success")
    
    if dcm2nii_out==0:
        print(get_proc_id(test), " worked")
        return 0
    else: #if dcm2nii produces error, remove all the output files
        print("Remove output files")
        rm_files = [f for f in os.listdir(nii_out_path) if f.startswith(sid)]
        for rm_file in rm_files:
            remove_file(rm_file)
            return 1 
        