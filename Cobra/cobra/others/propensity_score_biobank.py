import pandas as pd 
import numpy as np

data_folder = '/home/neus/Documents/09.UCPH/MasterThesis/DATA/feature_importance/interesting_for_CoBra'
means_file = f'{data_folder}/demografic_clinical_characteristics_ukbiobank.csv'
ors_file = f'{data_folder}/feature_importances_cmb_ukbiobank.csv'

means_df = pd.read_csv(means_file)
ors_df = pd.read_csv(ors_file)

prevalence = 0.07

#cohort size
n_population = means_df[ means_df['feature']=='n_population']
n_absent = n_population['mean_absent'].values[0]
n_present = n_population['mean_present'].values[0]

frac_absent = n_absent/(n_absent+n_present)
frac_present = n_present/(n_absent+n_present)

table = ors_df.merge(means_df,on='feature',how='inner',validate='one_to_one')

log_reg_intercept = np.log(prevalence/(1-prevalence)) - np.sum( np.log(table['eor'])*( frac_absent*table['mean_absent'] + frac_present*table['mean_present']) )

print(f'Logistic regression for CMB prediction using UK Biobank data. \n Intercept parameter, beta0 = {log_reg_intercept} \t exp(beta0)={np.exp(log_reg_intercept)}')
