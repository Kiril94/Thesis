# -*- coding: utf-8 -*-
"""
Created on Mon Nov  15 12:06:00 2021

@author: neusRodeja
"""
import sys
sys.path.insert(0, '/home/neus/Documents/09.UCPH/MasterThesis/github/Thesis/Cobra/cobra/')

import numpy as np
import pandas as pd 
from utilities.utils import load_scan_csv,df_unique_values,find_n_slices,save_nscans
from stats_tools.vis import create_1d_hist,create_2d_hist,create_boxplot
import matplotlib.pyplot as plt 

label = 'swi_neg'

main_folder = "/home/neus/Documents/09.UCPH/MasterThesis/github/Thesis/Cobra/cobra"
figs_folder = f'{main_folder}/figs/cmb_stats'
csv_folder = f"{main_folder}/tables"
csv_file_name = f"{label}_scans"

swi_pos_scans =  load_scan_csv(f'{csv_folder}/{csv_file_name}.csv')
n_scans = swi_pos_scans.shape[0]
n_patients = swi_pos_scans.drop_duplicates(subset='PatientID').shape[0]
print(f'Number of scans:\t{n_scans}\nNumber of patients:\t{n_patients}')


### STATISTICS ON FILTERED DATA ##################################################################
# Patients that: 
# - Have been tested positive before (or 3 days later) the scan
# - Have from 1 to 3 swi scans
# - From the patients with multiple scans, we have taken the last scan
#################################################################################################

n_scans = swi_pos_scans.shape[0]
n_patients = swi_pos_scans.drop_duplicates(subset='PatientID').shape[0]
print(f'AFTER FILTERING\nNumber of scans:\t{n_scans}\nNumber of patients:\t{n_patients}')

#Series Description
print(swi_pos_scans['SeriesDescription'].unique())

# Slice thickness distribution (Nominal slice thickness, in mm.)
thick_values = np.array(swi_pos_scans['SliceThickness'].dropna())
thick_min,thick_max = -0.5,11.5
fig,ax = plt.subplots()
ax.set(ylabel='Counts', xlabel='Slice Thickness [mm]')
create_1d_hist(ax,thick_values,12,(thick_min,thick_max),f'Slice thickness distribution for {label}',display_counts=True)
fig.savefig(f'{figs_folder}/{label}_sliceThickness.png')

#Spacing between slices distribution 
#(Spacing between slices, in mm. The spacing is measured from the center-to-center of each slice.)
spacing_values = np.array(swi_pos_scans['SpacingBetweenSlices'].dropna())
spacing_min,spacing_max = -0.5,10.5
fig,ax = plt.subplots()
ax.set(ylabel='Counts', xlabel='Spacing between slices (center-to-center) [mm]')
create_1d_hist(ax,spacing_values,11,(spacing_min,spacing_max),f'Spacing between slices distribution for {label}',display_counts=True)
fig.savefig(f'{figs_folder}/{label}_SpacingBetweenSlices.png')

#Uncovered space
#(Spacing between slices - Slice thickness)
uncovered_space = (swi_pos_scans['SpacingBetweenSlices']-swi_pos_scans['SliceThickness']).dropna()
uncov_min,uncov_max = np.min(uncovered_space),np.max(uncovered_space)
fig,ax = plt.subplots()
ax.set(ylabel='Counts', xlabel='(Spacing between slices - Slice thickness) [mm]')
create_1d_hist(ax,uncovered_space,9,(uncov_min-0.5,uncov_max+0.5),f'Uncovered/Overlaped space for {label}',display_counts=True)
fig.savefig(f'{figs_folder}/{label}_uncoveredSpce.png')

#Do the patients with overlap 1 mm have the same resolution
mask = ( (swi_pos_scans['SliceThickness']>=1.5) & (swi_pos_scans['SliceThickness']<2.5) & (swi_pos_scans['SpacingBetweenSlices']>=0.5) & (swi_pos_scans['SpacingBetweenSlices']<1.5) )
print(f"N.patients with Spacing between slices = 1 and Slice thickness = 2 : \t {np.shape(swi_pos_scans[mask])[0]}")

#Pixel spacing distribution 
#Physical distance in the patient between the center of each pixel
px_spacing_values = swi_pos_scans['PixelSpacing'].dropna()
px_spacing_values = list(filter(lambda x: len(x)>0,px_spacing_values))
px_spacing_x = np.array([float(px[0].split(' ')[0][1:]) for px in px_spacing_values])
px_spacing_y = np.array([float(px[0].split(' ')[1][:-1]) for px in px_spacing_values])

x_min,x_max=np.min(px_spacing_x),np.max(px_spacing_x)
y_min,y_max=np.min(px_spacing_y),np.max(px_spacing_y)

fig,ax = plt.subplots()
ax.set(xlabel='Row spacing [mm]', ylabel='Column spacing [mm]')
create_2d_hist(ax,px_spacing_x,px_spacing_y,[12,12],[[0.1,1.1],[0.1,1.1]],title='Pixel spacing distribution')
fig.savefig(f'{figs_folder}/{label}_pixelSpacing.png')


##Magnetic field strength
b_values = np.array(swi_pos_scans['MagneticFieldStrength'].dropna())
b_min,b_max = np.min(b_values),np.max(b_values)
fig,ax = plt.subplots()
ax.set(ylabel='Counts', xlabel=f'B$_0$ [T]')
create_1d_hist(ax,b_values,4,(1.25,3.25),f'Magnetic Field Strength for {label}',display_counts=True)
fig.savefig(f'{figs_folder}/{label}_magneticFieldStrength.png')

##Echo time
b_values = np.array(swi_pos_scans['EchoTime'].dropna())
b_min,b_max = np.min(b_values),np.max(b_values)
fig,ax = plt.subplots()
ax.set(ylabel='Counts', xlabel=f'time [ms]')
create_1d_hist(ax,b_values,6,(-5,55),f'Echo Time for {label}',display_counts=True)
fig.savefig(f'{figs_folder}/{label}_echoTime.png')

#Box plot for dimensions
swi_pos_scans.drop(index=2) #No slices in the folder????
 
x_dim = swi_pos_scans['Rows']
y_dim = swi_pos_scans['Columns']
z_dim = swi_pos_scans['NumberOfSlices']

fig,ax=plt.subplots(1,2,figsize=(12,5))
ax = ax.flatten()

ax[0] = create_boxplot(ax[0],[x_dim,y_dim,z_dim],['Height','Width','Depth'],title=f'Dimensions for {label} images')
ax[0].set(ylabel='#Pixels')

ax[1] = create_boxplot(ax[1],[px_spacing_x,px_spacing_y,spacing_values],['Pix.Spacing x', 'Pix.spacing y', 'Spacing Between Slices'],title='Spacing for SWI images')
ax[1].set(ylabel='Spacing [mm]')
fig.savefig(f'{figs_folder}/{label}_dimensions.png')

fig,ax=plt.subplots(2,2,figsize=(8,7),gridspec_kw={'width_ratios': [3, 1]})
ax = ax.flatten()

ax[0] = create_boxplot(ax[0],[x_dim,y_dim],['Height','Width'])
ax[0].set(ylabel='#Pixels')

ax[1] = create_boxplot(ax[1],[z_dim],['Depth'])
ax[1].set(ylabel='#Pixels')

ax[2] = create_boxplot(ax[2],[px_spacing_x,px_spacing_y],['Pix.Spacing x', 'Pix.spacing y'])
ax[2].set(ylabel='Spacing [mm]')

ax[3] = create_boxplot(ax[3],[spacing_values],['Spacing Between Slices'])
ax[3].set(ylabel='Spacing [mm]')

fig.savefig(f'{figs_folder}/{label}_dimensions_v2.png')

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





