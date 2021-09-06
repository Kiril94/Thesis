#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 12:02:58 2021

@author: neus
"""
'''USAGE
python read_metadata.py ROOT_DIR PATIENT_CSV_FILE SCAN_CSV_FILE
'''
import csv
import os 
import sys
import glob
import time
import data_access.load_data_tools as dt
from utilss.utils import get_running_time

folder = '2019_01'
start = time.time()

#set path
# main_path = sys.argv[1]
# main_path = '/home/neus/Documents/09.UCPH/MasterThesis/ucph_sif_data/2019_02'
main_path = '/run/user/1000/gvfs/sftp:host=sif-io.erda.dk/' + folder

try: 
    patient_file_name = sys.argv[2]
    
except IndexError:
    patient_file_name = './patient_table_'+folder+'.csv'
    print(f'Table PATIENT will be saved at the current folder as {patient_file_name:s}')
    pass

try: 
    scan_file_name = sys.argv[3]
    
except IndexError:
    scan_file_name = './scan_table_'+folder+'.csv'
    print(f'Table SCAN will be saved at the current folder as {scan_file_name:s}')
    pass

#Get patients directories 
patients_dir = glob.iglob(main_path+'/*')    
patients_dir_list = [x for x in patients_dir if os.path.isdir(x)] 


try:
    
    i = 1 #Counting patients to print in console
    
    scan_file = open(scan_file_name,'w', encoding = 'UTF8')    
    scan_writer = csv.writer(scan_file)
    scan_writer.writerow([x[0] for x in dt.get_scan_key_list()])
        
    patient_file = open(patient_file_name, 'w', encoding = 'UTF8')
    patient_writer = csv.writer(patient_file)
    patient_writer.writerow([x[0] for x in dt.get_patient_key_list()])
    
    for patient_dir in patients_dir_list:

        #Print information to safe track
        print('Patient %d out of %d %s' %(i, len(patients_dir_list),get_running_time(start)))
        
        patient_instance = dt.Patient(patient_dir)    
        patient_scans = patient_instance.all_scan_dicts(reconstruct_3d=False)    
            
        try:
            patient_writer.writerow(patient_instance.info().values())     
        except FileNotFoundError as fne:  
            print(fne)
        
        for scan in patient_scans:
            scan_writer.writerow(scan.values())
            
        i+=1

except:
    raise
    
finally:

    patient_file.close()
    scan_file.close()
    #record time
    print('Elapsed time: %s' %(get_running_time(start)))

