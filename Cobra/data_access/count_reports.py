#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 17:27:35 2021

@author: neus
"""
from glob import iglob
import sys
import time 
from utilss.utils import get_running_time

#### USAGE
# python count_reports.py FOLDER_NAME BASE_DIR FILES_DIR 


def get_docs_path_list(scan_dir):
    reports = iglob(f"{scan_dir}/*/*/DOC/*/*.pdf")
    reports_list = [x for x in reports] 
    return reports_list

def count_docs(scan_dir):
    n_reports = sum([1 for x in iglob(f"{scan_dir}/*/*/DOC/*/*.pdf")] )
    return n_reports

start = time.time()
try:
    folder_name = sys.argv[1]
except IndexError:
    folder_name = '2019_12'
    
try:
    scan_dir = sys.argv[2]
except IndexError:
    #scan_dir = '/home/neus/Documents/09.UCPH/MasterThesis/ucph_sif_data/2019_02'
    scan_dir = '/home/neus/sif/'+folder_name

try: 
    files_dir = sys.argv[3]
except IndexError:
    files_dir = '/home/neus/Documents/09.UCPH/MasterThesis'
    
    
n_reports = count_docs(scan_dir)
L = f'{n_reports:d} in {scan_dir:s}. Running time: {get_running_time(start):s}'
print(L)


file = open(files_dir+'/n_reports_'+folder_name+'.txt','w')
file.write(L)
file.close()

## 3383 in 2019_01
## 3035 in 2019_02
## 2944 in 2019_03
## 3056 in 2019_04
## 3331 in 2019_05
## 2799 in 2019_06
## 2187 in 2019_07
## 2829 in 2019_08
## 2767 in 2019_09
## 2165 in 2019_10
## 2888 in 2019_11
## in 2019_12
## 1402 in positive

## 32786 in total