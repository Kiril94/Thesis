"""Calculation from other studies.

Created on Nov 19th 2021

@author: Neus Rodeja Ferrer
"""
import sys
sys.path.insert(0, '/home/neus/Documents/09.UCPH/MasterThesis/github/Thesis/Cobra/cobra/')

from utilities.stats import weighted_mean,weighted_std
from utilities.utils import get_MDE
import numpy as np

#Multiple sclerosis Case-Control study

#Values for all ages
cmbs_number_means_all = np.array([1.4,1.5,1.,1.2])
cmbs_number_stds_all = np.array([0.7,0.8,0.,0.4])

cmbs_volume_means_all = np.array([36.3,42.3,32.3,21.1])
cmbs_volume_stds_all = np.array([40.5,38.0,26.6,15.8])

weights_all = np.array([51,6,5,9])

cmbs_number_mean_all = weighted_mean(cmbs_number_means_all,weights_all)
cmbs_number_std_all = weighted_std(cmbs_number_stds_all,weights_all)

cmbs_volume_mean_all = weighted_mean(cmbs_volume_means_all,weights_all)
cmbs_volume_std_all = weighted_std(cmbs_volume_stds_all,weights_all)

print(f'For all population\n# CMBs = \t({cmbs_number_mean_all:.2f}+\-{cmbs_number_std_all:.2f}) units \nCMBs volume = \t({cmbs_volume_mean_all:.2f}+\-{cmbs_volume_std_all:.2f}) mm³')
#Values for <= 50 y.o.
cmbs_number_means_50 = np.array([1.4,1.3,1.3])
cmbs_number_stds_50 = np.array([0.7,0.5,0.5])

cmbs_volume_means_50 = np.array([37.1,31.5,25.5])
cmbs_volume_stds_50 = np.array([44.,38.4,17.8])

weights_50 = np.array([41,4,6])

cmbs_number_mean_50 = weighted_mean(cmbs_number_means_50,weights_50)
cmbs_number_std_50 = weighted_std(cmbs_number_stds_50,weights_50)

cmbs_volume_mean_50 = weighted_mean(cmbs_volume_means_50,weights_50)
cmbs_volume_std_50 = weighted_std(cmbs_volume_stds_50,weights_50)

print(f'For >=50 y.o. population\n# CMBs = \t({cmbs_number_mean_50:.2f}+\-{cmbs_number_std_50:.2f}) units \nCMBs volume = \t({cmbs_volume_mean_50:.2f}+\-{cmbs_volume_std_50:.2f}) mm³')

#SAME excluding MS
cmbs_number_mean_all = weighted_mean(cmbs_number_means_all[1:],weights_all[1:])
cmbs_number_std_all = weighted_std(cmbs_number_stds_all[1:],weights_all[1:])

cmbs_volume_mean_all = weighted_mean(cmbs_volume_means_all[1:],weights_all[1:])
cmbs_volume_std_all = weighted_std(cmbs_volume_stds_all[1:],weights_all[1:])

cmbs_number_mean_50 = weighted_mean(cmbs_number_means_50[1:],weights_50[1:])
cmbs_number_std_50 = weighted_std(cmbs_number_stds_50[1:],weights_50[1:])

cmbs_volume_mean_50 = weighted_mean(cmbs_volume_means_50[1:],weights_50[1:])
cmbs_volume_std_50 = weighted_std(cmbs_volume_stds_50[1:],weights_50[1:])

print('\n\nEXCLUDING PATIENTS WITH MS')
print(f'For all population\n# CMBs = \t({cmbs_number_mean_all:.2f}+\-{cmbs_number_std_all:.2f}) units \nCMBs volume = \t({cmbs_volume_mean_all:.2f}+\-{cmbs_volume_std_all:.2f}) mm³')
print(f'For >=50 y.o. population\n# CMBs = \t({cmbs_number_mean_50:.2f}+\-{cmbs_number_std_50:.2f}) units \nCMBs volume = \t({cmbs_volume_mean_50:.2f}+\-{cmbs_volume_std_50:.2f}) mm³')

###Do the power calculation 

power = 0.8
significance = 0.05
P = 124/366
N = 124
std_number = 0.5
mde_number = get_MDE(std_number,N,power,significance,P)
std_volume = 27
mde_volume = get_MDE(std_volume,N,power,significance,P)

print(f'\n\nMDE={mde_number} #CMBs \nMDE={mde_volume} mm³')