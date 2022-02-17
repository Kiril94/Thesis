"""
created on Wed 16th Feb 2020
author: Neus Rodeja Ferrer
"""
import pandas as pd

original_file = "/home/neus/Documents/09.UCPH/MasterThesis/github/Thesis/Cobra/cobra/tables/SWIMatching/d_cohen_propensity_covid_0.csv"
output_file = f'{original_file.split(".")[0]}.{original_file.split(".")[1]}_male_latex.{original_file.split(".")[-1]}'

df = pd.read_csv(original_file)
df = df[ ( (df['is_matched']==1)&(df['is_male']==1)&(df['is_general']==0) ) ]
print(df.keys())

string_columns = ['feature']
int_columns = ['cases_exposed', 'cases_unexposed', 'controls_exposed', 'controls_unexposed', 'is_matched', 'is_general', 'is_male' ]
float_columns = [ 'd', 'sigma']

for key in string_columns:
    df[key] = df[key].str.replace('_',' ')

for key in int_columns:
    df[key] = df[key].astype(int)

for key in float_columns:
    df[key] =  df[key].map('{:,.5f}'.format)

df.to_csv(output_file,index=False)
print(f'Saved in {output_file}')