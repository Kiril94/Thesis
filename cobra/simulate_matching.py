#%%
# Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utilities import matching as mat
from utilities import matching
import importlib
from numpy.random import default_rng
from scipy.stats import norm
importlib.reload(matching)
pd.options.mode.chained_assignment = None
#%% 
# Test

#%%
# Specify params
num_variables = 5
population_size = 10000
num_hidden_variables = 1
odds_exposed = .1
true_OR = 20
random_state = 0
#%%
# Simulate exposure
rng = np.random.RandomState(0)
loc = list(rng.rand(num_variables)*.5)
loc.insert(0, 4)
beta = norm.rvs(loc=loc, scale=1,size=num_variables+1, 
            random_state=random_state)
df = matching.simulate_exposure(beta, num_variables, num_hidden_variables,
     population_size, random_state=random_state)
print('Number exposed', df.exposed.sum())
df.head()
#%%
# Simulate disease
df = matching.simulate_disease(df, odds_exposed, true_OR)
df = df.astype({'disease': 'int32'})
df = df.astype({'exposed': 'int32'})
ct = matching.get_contingency_table(df)
matching.plot_heatmap(ct)
OR, CI, pval = matching.compute_OR_CI_pval(df, print_=True, start_string='Estimated from whole population')
#%%
matching.plot_variables_kde(df)
#%%
# Select random subset
dfd = df[df.disease==1]
dfnd = df[df.disease==0]
n_subset = 500
deck = np.arange(n_subset)
rng = default_rng(random_state)
rng.shuffle(deck)
df_rand = dfnd.iloc[deck, :].reset_index()
df_subs = pd.concat([dfd, df_rand], ignore_index=True)
OR_nm, CI_nm, pval_nm = matching.compute_OR_CI_pval(df_subs, print_=True, 
    start_string='No matching')

#%%
# Estimate PS
df_subs = matching.estimate_PS(df_subs)
sns.histplot(data=df_subs, x='PS', bins=50, hue='exposed')
df.head()
#%%
from sklearn.neighbors import NearestNeighbors
def NN_matching(df):
    df_cases = df[df.disease==1]
    PS_cases = df_cases.PS.to_numpy()
    df_controls = df[df.disease==0]
    PS_controls = df_controls.PS.to_numpy()
    neigh = NearestNeighbors(n_neighbors=10)
    neigh.fit(PS_controls.reshape(-1,1))
    distances, indices = neigh.kneighbors(PS_cases.reshape(-1,1))
    indices = indices.flatten()
    df_matched_controls = df_controls.iloc[indices, :]
    df = pd.concat([df_cases, df_matched_controls], ignore_index=True)
    return df

df_subs_m = NN_matching(
    df_subs)
OR_nm, CI_nm, pval_nm = matching.compute_OR_CI_pval(df_subs_m, print_=True, 
    start_string='No matching')
ct = matching.get_contingency_table(df_subs_m)
matching.plot_heatmap(ct)
#e_PS_matched_controls = e_PS_controls[indices.flatten()]
#x_matched_controls = x_controls[indices.flatten()]
