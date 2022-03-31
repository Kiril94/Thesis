"""
created on march 29th 2020
author: Neus Rodeja Ferrer
"""

#%%
import numpy as np
from pathlib import Path
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import join

main_path = Path(__file__).parent.parent
tables_path = main_path / "tables"
figs_path = main_path / "figs" / "cmb_stats" / "matching_v3"
included_swi_file = tables_path / "SWIMatching" / "ids_swi_included_v3.csv"
excluded_swi_file = tables_path / "SWIMatching" / "ids_swi_excluded_v3.csv"
all_swi_file = tables_path / "swi_all.csv"
probs_swi = tables_path / "SWIMatching" / "ids_swi_excluded_pcmb_v3.csv"


included_ids = pd.read_csv(included_swi_file)
excluded_ids = pd.read_csv(excluded_swi_file)
info_df = pd.read_csv(all_swi_file)
probs_exc = pd.read_csv(probs_swi)

included_info = included_ids.merge(info_df,how='inner',on='PatientID',validate='one_to_one')
excluded_info = excluded_ids.merge(info_df,how='inner',on='PatientID',validate='one_to_one')
excluded_info = excluded_info.merge(probs_exc,how='inner',on='PatientID',validate='one_to_one')
excluded_info.sort_values(by='p_cmb',inplace=True,ascending=False)
#%%
#count how many are converted
converted = pd.read_csv(join("F:","CoBra","Data","swi_nii","converted_excluded.csv"))
high_excl = excluded_info.head(50)
high_excl['PatientID'].to_csv(tables_path/"SWIMatching"/"25_high_excluded.csv",index=False)
conv_high_excl = high_excl[ high_excl['PatientID'].isin(converted['PatientID'])]
¡
#%%
#box plots for dimensions

def create_boxplot(ax,data,data_labels=None,title=''):
    
    if (data_labels is None):
        data_labels = [int(i+1) for i in range(len(data))]
    ax = sns.boxplot(ax=ax,data=data,palette='Set2',medianprops=dict(color="red"))
    ax = sns.stripplot(ax=ax,data=data, color=".25",alpha=0.3)
    ax.set_xticklabels(data_labels)
    ax.get_xaxis().tick_bottom()
    ax.set_title(title)
    
    return ax

x_dim = included_info['Rows']
y_dim = included_info['Columns']
z_dim = included_info['NumberOfSlices']
px_spacing_x = included_info['RowSpacing']
px_spacing_y = included_info['ColumnSpacing']
spacing_values = included_info['SpacingBetweenSlices'] #.dropna()

fig,ax=plt.subplots(1,2,figsize=(12,5))
ax = ax.flatten()

ax[0] = create_boxplot(ax[0],[x_dim,y_dim,z_dim],['Height','Width','Depth'],title=f'Dimensions for included scans')
ax[0].set(ylabel='#Pixels')

ax[1] = create_boxplot(ax[1],[px_spacing_x,px_spacing_y,spacing_values],['Pix.Spacing x', 'Pix.spacing y', 'Spacing Between Slices'],title='Spacing for SWI images')
ax[1].set(ylabel='Spacing [mm]')

fig.suptitle("Scans included in CMB study")
fig.savefig(f'{figs_path}/included_boxplot_dimensions.png')

#for excluded
x_dim = excluded_info['Rows']
y_dim = excluded_info['Columns']
z_dim = excluded_info['NumberOfSlices']
px_spacing_x = excluded_info['RowSpacing']
px_spacing_y = excluded_info['ColumnSpacing']
spacing_values = excluded_info['SpacingBetweenSlices'] #.dropna()

fig,ax=plt.subplots(1,2,figsize=(12,5))
ax = ax.flatten()

ax[0] = create_boxplot(ax[0],[x_dim,y_dim,z_dim],['Height','Width','Depth'],title=f'Dimensions for excluded scans')
ax[0].set(ylabel='#Pixels')

ax[1] = create_boxplot(ax[1],[px_spacing_x,px_spacing_y,spacing_values],['Pix.Spacing x', 'Pix.spacing y', 'Spacing Between Slices'],title='Spacing for SWI images')
ax[1].set(ylabel='Spacing [mm]')

fig.suptitle("Scans excluded in CMB study")
fig.savefig(f'{figs_path}/excluded_boxplot_dimensions.png')

#%% 
#inc manufacturers

manufacturer_models_name = included_info['ManufacturerModelName'].unique()

groups_manufacturer = included_info.groupby('ManufacturerModelName')
fig,ax = plt.subplots()
ax = groups_manufacturer.size().plot.bar()
fig.savefig(f"{figs_path}/included_manufacturers.png")

#%% excl manu
manufacturer_models_name = excluded_info['ManufacturerModelName'].unique()

groups_manufacturer = excluded_info.groupby('ManufacturerModelName')
fig,ax = plt.subplots()
ax = groups_manufacturer.size().plot.bar()
fig.savefig(f"{figs_path}/excluded_manufacturers.png")

#%%
# for 25 higher probs
high_exc = excluded_info.head(25)
#for excluded
x_dim = high_exc['Rows']
y_dim = high_exc['Columns']
z_dim = high_exc['NumberOfSlices']
px_spacing_x = high_exc['RowSpacing']
px_spacing_y = high_exc['ColumnSpacing']
spacing_values = high_exc['SpacingBetweenSlices'] #.dropna()

fig,ax=plt.subplots(1,2,figsize=(12,5))
ax = ax.flatten()

ax[0] = create_boxplot(ax[0],[x_dim,y_dim,z_dim],['Height','Width','Depth'],title=f'Dimensions for excluded scans')
ax[0].set(ylabel='#Pixels')

ax[1] = create_boxplot(ax[1],[px_spacing_x,px_spacing_y,spacing_values],['Pix.Spacing x', 'Pix.spacing y', 'Spacing Between Slices'],title='Spacing for SWI images')
ax[1].set(ylabel='Spacing [mm]')

fig.suptitle("Scans excluded (25 with highest P-cmb) in CMB study")



groups_manufacturer = high_exc.groupby('ManufacturerModelName')
fig,ax = plt.subplots()
ax = groups_manufacturer.size().plot.bar()
fig.suptitle("Scans excluded (25 with highest P-cmb) in CMB study")
#fig.savefig(f'{figs_path}/excluded_boxplot_dimensions.png')
