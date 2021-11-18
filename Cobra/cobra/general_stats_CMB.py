# -*- coding: utf-8 -*-
"""
Created on Mon Nov  15 12:06:00 2021

@author: neusRodeja
"""

import numpy as np
import pandas as pd 
from utilities.utils import load_scan_csv,df_unique_values,find_n_slices,save_nscans
from stats_tools.vis import create_1d_hist,create_2d_hist,create_boxplot
import matplotlib.pyplot as plt 


main_folder = "/home/neus/Documents/09.UCPH/MasterThesis/github/Thesis/Cobra/cobra"
figs_folder = f'{main_folder}/figs/cmb_stats'
csv_folder = f"{main_folder}/tables"
csv_file_name = "swi_pos_scans"
# csv_file_name = "neg_pos_clean"

# table_all = load_scan_csv(f'{csv_folder}/{csv_file_name}.csv')
# ###### Inspect quality for SWI positive scans.
# mask_nan_values =  (table_all['InstanceCreationDate'].notna()&table_all['DateTime'].notna())
# swi_pos_scans = table_all[(table_all['Positive']==1)&(table_all['Sequence']=='swi')&(table_all['days_since_test']>-3)&mask_nan_values]
# #swi_pos_patients = swi_pos_scans.drop_duplicates(substet=['PatientID'])

# swi_pos_scans = save_nscans(swi_pos_scans,f'{csv_folder}/{swi_pos_scans}')

swi_pos_scans =  load_scan_csv(f'{csv_folder}/{csv_file_name}.csv')
n_scans = swi_pos_scans.shape[0]
n_patients = swi_pos_scans.drop_duplicates(subset='PatientID').shape[0]
print(f'Number of scans:\t{n_scans}\nNumber of patients:\t{n_patients}')

#Box plot
swi_pos_scans.drop(index=2) #No slices in the folder????

x_dim = swi_pos_scans['Rows']
y_dim = swi_pos_scans['Columns']
z_dim = swi_pos_scans['NumberOfSlices']
data_to_plot = [x_dim,y_dim,z_dim]
data_labels = ['dim x','dim y','dim z']

fig,ax=plt.subplots()
ax = create_boxplot(ax,data_to_plot,data_labels,title='Dimensions for SWI images')
ax.set(ylabel='#Pixels')
fig.savefig(f'{figs_folder}/swi_pos_dimensions.png')


scans_groupedbyPatient = swi_pos_scans.groupby(['PatientID'])
# Scans per patient distribution
scans_perPatient = scans_groupedbyPatient.size()
scans_min,scans_max = np.min(scans_perPatient),np.max(scans_perPatient)
fig,ax = plt.subplots()
ax.set(ylabel='Counts',xlabel='#Scans per patient')
hist_data = create_1d_hist(ax,scans_perPatient,14,(-0.5,13.5),'Positive patients with SWI scans',display_counts=True)
fig.savefig(f'{figs_folder}/swi_scans_per_patient.png')

#Take patients that have only 1,2 or 3 scans
patients_id = scans_perPatient[scans_perPatient<4].keys()
swi_pos_scans = swi_pos_scans[swi_pos_scans['PatientID'].isin(patients_id)]
swi_pos_scans = swi_pos_scans.sort_values('days_since_test',ascending=False)

#Days since test for patients with multiple scans
swi_pos_scans = swi_pos_scans.groupby('PatientID').first().reset_index() ###TAKING THE LATEST SCAN 
days = swi_pos_scans['days_since_test']
days_min,days_max = np.min(days),np.max(days)
fig,ax = plt.subplots()
ax.set(ylabel='Counts',xlabel='#Days since positive test')
create_1d_hist(ax,days,100,(days_min,days_max),'Postive patients with multiple SWI')
fig.savefig(f'{figs_folder}/swi_days_since_test.png')


#print(swi_pos_scans.iloc[0])

### STATISTICS ON FILTERED DATA ##################################################################
# Patients that: 
# - Have been tested positive before (or 3 days later) the scan
# - Have from 1 to 3 swi scans
# - From the patients with multiple scans, we have taken the last scan
#################################################################################################

n_scans = swi_pos_scans.shape[0]
n_patients = swi_pos_scans.drop_duplicates(subset='PatientID').shape[0]
print(f'AFTER FILTERING\nNumber of scans:\t{n_scans}\nNumber of patients:\t{n_patients}')

# # Slice thickness distribution (Nominal slice thickness, in mm.)
# thick_values = np.array(swi_pos_scans['SliceThickness'].dropna())
# thick_min,thick_max = -0.5,11.5
# fig,ax = plt.subplots()
# ax.set(ylabel='Counts', xlabel='Slice Thickness [mm]')
# create_1d_hist(ax,thick_values,12,(thick_min,thick_max),'Slice thickness distribution for positive SWI',display_counts=True)
# fig.savefig(f'{figs_folder}/swi_pos_sliceThickness.png')

# #Spacing between slices distribution 
# #(Spacing between slices, in mm. The spacing is measured from the center-to-center of each slice.)
# spacing_values = np.array(swi_pos_scans['SpacingBetweenSlices'].dropna())
# spacing_min,spacing_max = -0.5,10.5
# fig,ax = plt.subplots()
# ax.set(ylabel='Counts', xlabel='Spacing between slices (center-to-center) [mm]')
# create_1d_hist(ax,spacing_values,11,(spacing_min,spacing_max),'Spacing between slices distribution for positive SWI',display_counts=True)
# fig.savefig(f'{figs_folder}/swi_pos_SpacingBetweenSlices.png')

# #Uncovered space
# #(Spacing between slices - Slice thickness)
# uncovered_space = (swi_pos_scans['SpacingBetweenSlices']-swi_pos_scans['SliceThickness']).dropna()
# uncov_min,uncov_max = np.min(uncovered_space),np.max(uncovered_space)
# fig,ax = plt.subplots()
# ax.set(ylabel='Counts', xlabel='(Spacing between slices - Slice thickness) [mm]')
# create_1d_hist(ax,uncovered_space,9,(uncov_min-0.5,uncov_max+0.5),'Uncovered/Overlaped space for positive SWI',display_counts=True)
# fig.savefig(f'{figs_folder}/swi_pos_uncoveredSpce.png')

# #Do the patients with overlap 1 mm have the same resolution
# mask = ( (swi_pos_scans['SliceThickness']>=1.5) & (swi_pos_scans['SliceThickness']<2.5) & (swi_pos_scans['SpacingBetweenSlices']>=0.5) & (swi_pos_scans['SpacingBetweenSlices']<1.5) )
# print(f"N.patients with Spacing between slices = 1 and Slice thickness = 2 : \t {np.shape(swi_pos_scans[mask])[0]}")

# #Pixel spacing distribution 
# #Physical distance in the patient between the center of each pixel
# px_spacing_values = swi_pos_scans['PixelSpacing'].dropna()
# px_spacing_x = np.array([px[0] for px in px_spacing_values])
# px_spacing_y = np.array([px[1] for px in px_spacing_values])

# x_min,x_max=np.min(px_spacing_x),np.max(px_spacing_x)
# y_min,y_max=np.min(px_spacing_y),np.max(px_spacing_y)

# fig,ax = plt.subplots()
# ax.set(xlabel='Row spacing [mm]', ylabel='Column spacing [mm]')
# create_2d_hist(ax,px_spacing_x,px_spacing_y,[12,12],[[0.1,1.1],[0.1,1.1]],title='Pixel spacing distribution')
# fig.savefig(f'{figs_folder}/swi_pos_pixelSpacing.png')


# ##Magnetic field strength
# b_values = np.array(swi_pos_scans['MagneticFieldStrength'].dropna())
# b_min,b_max = np.min(b_values),np.max(b_values)
# fig,ax = plt.subplots()
# ax.set(ylabel='Counts', xlabel=f'B$_0$ [T]')
# create_1d_hist(ax,b_values,4,(1.25,3.25),'Magnetic Field Strength for positive SWI',display_counts=True)
# fig.savefig(f'{figs_folder}/swi_pos_magneticFieldStrength.png')

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





