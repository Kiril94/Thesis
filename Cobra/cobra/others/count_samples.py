"""
created on 21st Feb 2022
author: neus Rodeja Ferrer
"""
import pandas as pd 
import numpy as np

all_info = pd.read_csv('/home/neus/Documents/09.UCPH/MasterThesis/github/Thesis/Cobra/cobra/tables/SynthCMB/all_info.csv')
samples_file = open('/home/neus/Documents/09.UCPH/MasterThesis/github/Thesis/Cobra/cobra/tables/SynthCMB/images_list.txt','r')
lines = samples_file.readlines()

h,w = 176,256

# n_background = 0
# n_foreground = 0
# for line in lines:
#      parts = line.split('_')
#      file_name = '_'.join(parts[:-1]) + '.nii.gz' 
#      z = int(parts[-1][5:-8])

#      cmbs = all_info[ (all_info['NIFTI File Name']==file_name)&(all_info['z_position']==z) ]
#      n_cmbs = len(cmbs)
     
#      if (n_cmbs == 0):
#          'IMAGE WITH NO CMB'
#          print(file_name)

#      n_foreground += 9*n_cmbs
#      n_background += h*w - 9*n_cmbs


n_background = 1101058279
n_foreground = 245529

def compute_effective_weights(n_samples,beta=0.999999):
    
    weights = (1-beta**np.array(n_samples) )/(1-beta)
    weights = (1/weights)
    weights = weights/(np.sum(weights))

    return weights

weights = compute_effective_weights([n_background,n_foreground])

print(f'Size of training set \t{len(lines)}')
print(f'Sample sizes: BACKGROUND\t{n_background} FOREGROUND\t{n_foreground}')
print(f'Weights: BACCKGROUND\t{weights[0]} FOREGROUND\t{weights[1]}')
