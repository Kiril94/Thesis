#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 12:02:58 2021

@author: neus
"""
'''USAGE
python read_metadata.py ROOT_DIR PATIENT_CSV_FILE SCAN_CSV_FILE
'''
import pandas as pd
import os 
import sys
import glob
import time
import load_data_tools as dt

start = time.time()

#Define 2 tables
all_scans = [] 
all_patients = []

#set path
main_path = sys.argv[1]
# main_path = '/home/neus/Documents/09.UCPH/MasterThesis/ucph_sif_data'

try: 
    patient_file = sys.argv[2]
    
except IndexError:
    print('Table PATIENT will be saved at the current folder as patient_table.csv')
    patient_file = './patient_table.csv'
    pass

try: 
    scan_file = sys.argv[3]
    
except IndexError:
    print('Table SCAN will be saved at the current folder as scan_table.csv')
    scan_file = './scan_table.csv'
    pass

#Get patients directories 
patients_dir = glob.iglob(main_path+'/*')    
patients_dir_list = [x for x in patients_dir if os.path.isdir(x)] 


try:
    
    i = 1
    for patient_dir in patients_dir_list:
        m, s = divmod(time.time()-start , 60)
        h, m = divmod(m, 60)
        print('Patient %d out of %d [%d h %d m %d s]' %(i, len(patients_dir_list),h,m,s))
        i+=1
        patient_instance = dt.Patient(patient_dir)    
        patient_scans = patient_instance.all_scan_dicts(reconstruct_3d=False)    
        all_scans += patient_scans
        
        try:
            all_patients += [patient_instance.info()]        
        except FileNotFoundError as fne:  
            print(fne)

except:
    raise
    
finally:

    #save data
    table_patients = pd.DataFrame(all_patients)
    table_patients.to_csv(patient_file,index=False)
        
    table_scans = pd.DataFrame(all_scans)
    table_scans.to_csv(scan_file,index=False)
    
    #record time
    end = time.time()
    
    m, s = divmod(end-start , 60)
    h, m = divmod(m, 60)
    print('Elapsed time: %d h, %d m, %d s' %(h,m,s))

