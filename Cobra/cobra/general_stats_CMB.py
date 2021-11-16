# -*- coding: utf-8 -*-
"""
Created on Mon Nov  15 12:06:00 2021

@author: neusRodeja
"""

import numpy as np
import pandas as pd 
from utilities.utils import load_scan_csv,df_unique_values
from stats_tools.vis import create_1d_hist,create_2d_hist
import matplotlib.pyplot as plt 


main_folder = "/home/neus/Documents/09.UCPH/MasterThesis/github/Thesis/Cobra/cobra"
figs_folder = f'{main_folder}/figs/cmb_stats'
csv_folder = f"{main_folder}/tables"
csv_file_name = "neg_pos_clean"

table_all = load_scan_csv(f'{csv_folder}/{csv_file_name}.csv')
print(table_all)

###### Inspect quality for SWI positive scans. 
swi_pos_scans = table_all[(table_all['Positive']==1)&(table_all['Sequence']=='swi')&(table_all['days_since_test']>-3)]
#swi_pos_patients = swi_pos_scans.drop_duplicates(substet=['PatientID'])

# Slice thickness distribution
thick_values = np.array(swi_pos_scans['SliceThickness'].dropna())
thick_min,thick_max = -0.5,11.5
fig,ax = plt.subplots()
ax.set(ylabel='Counts')
create_1d_hist(ax,thick_values,12,(thick_min,thick_max),'Slice thickness distribution for positive SWI')
fig.savefig(f'{figs_folder}/swi_pos_sliceThickness.png')

#Spacing between slices distribution 
spacing_values = np.array(swi_pos_scans['SpacingBetweenSlices'].dropna())
spacing_min,spacing_max = -0.5,10.5
fig,ax = plt.subplots()
ax.set(ylabel='Counts')
create_1d_hist(ax,spacing_values,11,(spacing_min,spacing_max),'Spacing between slices distribution for positive SWI')
fig.savefig(f'{figs_folder}/swi_pos_SpacingBetweenSlices.png')

#Pixel spacing distribution 
px_spacing_values = swi_pos_scans['PixelSpacing'].dropna()
px_spacing_x = np.array([px[0] for px in px_spacing_values])
px_spacing_y = np.array([px[1] for px in px_spacing_values])

x_min,x_max=np.min(px_spacing_x),np.max(px_spacing_x)
y_min,y_max=np.min(px_spacing_y),np.max(px_spacing_y)

fig,ax = plt.subplots()
create_2d_hist(ax,px_spacing_x,px_spacing_y,[12,12],[[0.1,1.1],[0.1,1.1]],title='Pixel spacing distribution')
fig.savefig(f'{figs_folder}/swi_pos_pixelSpacing.png')

##Magnetic field strength
b_values = np.array(swi_pos_scans['MagneticFieldStrength'].dropna())
b_min,b_max = np.min(b_values),np.max(b_values)
fig,ax = plt.subplots()
ax.set(ylabel='Counts')
create_1d_hist(ax,b_values,4,(1.25,3.25),'Magnetic Field Strength for positive SWI')
fig.savefig(f'{figs_folder}/swi_pos_magneticFieldStrength.png')

#plt.show()


# columns = ['AngioFlag',
#        'AcquisitionMatrix', 'AcquisitionContrast', 'AcquisitionDuration',
#        'dBdt', 
#        #'EchoTime', 
#        'EchoTrainLength', 'EchoNumbers', 'FlipAngle',
#        #'FrameOfReferenceUID', 
#        'ImagingFrequency', 'ImagedNuclues', 'InversionTime',
#        'ImagesInAcquisition', 
#        #'ImageType', 
#        'MagneticFieldStrength',
#        'Manufacturer', 'ManufacturerModelName', 'MRAcquisitionType',
#        'NumberofAverages', 'NumberOfEchoes', 'NumberofPhaseEncodingSteps',
#        'PatientPosition', 'PixelBandwith', 'PixelPresentation', 
#        #'PixelSpacing',
#        'PhotometricInterpretation', 'PulseSequenceName', 'RepetitionTime',
#        #'Rows', 'Columns', 
#        #'ScanningSequence', 
#        #'SequenceVariant',
#        'SequenceName', 
#        #'ScanOptions', 
#        #'SeriesDescription', 
#        'SoftwareVersions',
#        'SliceThickness', 'StudyPriorityID', 'PatientPosition.1',
#        'SpacingBetweenSlices', 'SecondEcho', 'VariableFlipAngleFlag',
#        'DateTime', 'Sequence', 'TrueSequenceType', 'Positive',]





