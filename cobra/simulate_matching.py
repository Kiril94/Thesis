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
num_variables = 3
population_size = 10000
num_hidden_variables = 1
odds_exposed = .1
true_OR = 20
random_state = 0
#%%
# Simulate exposure
def get_rand_uniform(num_variables, random_state=0):
    rng = np.random.RandomState(random_state)
    return rng.rand(num_variables)

beta_loc = list(get_rand_uniform(num_variables)-.5)
beta_loc.insert(0,-4)
beta = norm.rvs(loc=beta_loc, scale=1, size=num_variables+1, 
            random_state=2)
fig, ax = plt.subplots(1,2)
ax[0].hist(beta[1:])
ax[0].set_xlabel('beta')
ax[0].set_ylabel('count')
df = matching.simulate_exposure(beta, num_variables, num_hidden_variables,
     population_size, random_state=random_state)
ax[1].scatter(df.x1, df.exposed)
ax[1].set_xlabel('x')
ax[1].set_ylabel('exposed')
fig. tight_layout()
print('Number exposed', df.exposed.sum())
matching.plot_variables_kde(df)

#%%
# test
def get_gamma1(OR, gamma0):
    """Calculate gamma in gamma*t from the OR (rare disease assumption)"""
    return np.log(OR)
gamma0 = 3
gamma1 = get_gamma1(true_OR, gamma0)
print(gamma1)
gamma_loc = list(get_rand_uniform(num_variables,2)+1)
gamma_loc.insert(0, gamma1)
gamma_loc.insert(0, gamma0)
gamma = norm.rvs(loc=gamma_loc, scale=1, size=num_variables+2, 
            random_state=random_state+1)

def compute_disease_proba(df, gamma):
    variables_cols = [col for col in df.keys() \
        if col.startswith('x') or col.startswith('hx')]
    disease_proba =  1/(1+np.exp(-(gamma[0]+gamma[1]*df.exposed\
        +gamma[2:]@df[variables_cols].T))
    df['disease_proba'] = disease_proba
    return df
def simulate_disease(df, random_state=0):
    if type(random_state)==int:
        rng = np.random.RandomState(random_state+2)
    else:
        rng = np.random
    df['disease'] = rng.rand(len(df))<df.disease_proba
    return df

df = compute_disease_proba(df, gamma)
dfd = simulate_disease(df)
ctd = matching.get_contingency_table(dfd)
matching.plot_heatmap(ctd)
OR, pval = matching.compute_OR_pval(dfd)
print(OR, pval)





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
#%%
#
np.log(20)