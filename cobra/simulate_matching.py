#%%
# Import
import time
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
import multiprocessing as mp
pd.options.mode.chained_assignment = None
#%% 
# Test

#%%
# Specify params
num_variables = 1
population_size = 5000
num_hidden_variables = 0
true_OR = 3
random_state = 0
#%%
# Simulate exposure


#beta_loc = list(get_rand_uniform(num_variables)-.5)
#beta_loc.insert(0, -1)
#beta = norm.rvs(loc=beta_loc, scale=1, size=num_variables+1, 
#            random_state=2)

fig, ax = plt.subplots()
#ax[0].hist(beta[1:])
#ax[0].set_xlabel('beta')
#ax[0].set_ylabel('count')
beta = np.array([0,2])
df = matching.simulate_exposure(beta, num_hidden_variables,
     population_size, random_state=random_state)
ax.scatter(df.x0, df.exposed)
ax.set_xlabel('x')
ax.set_ylabel('exposed')
fig. tight_layout()
print('Number exposed', df.exposed.sum())
#matching.plot_variables_kde(df)

#%%
# test
def get_gamma1(OR):
    """Calculate gamma in gamma*t from the OR (rare disease assumption)"""
    return np.log(OR)
gamma0 = -4
gamma1 = get_gamma1(true_OR)
gamma = np.array([gamma0, gamma1, .1])
#gamma_loc = list(get_rand_uniform(num_variables,2)*0.0001-2)
#gamma_loc.insert(0, gamma1)
#gamma_loc.insert(0, gamma0)
#gamma = norm.rvs(loc=gamma_loc, scale=.0001, 
#            size=num_variables+2, 
#            random_state=random_state+1)

def compute_disease_proba(df, gamma):
    variables_cols = [col for col in df.keys() \
        if col.startswith('x') or col.startswith('hx')]
    disease_proba =  1/(1+np.exp(-(gamma[0]+gamma[1]*df.exposed\
        +gamma[2:]@df[variables_cols].T)))
    df['disease_proba'] = disease_proba
    return df
df = compute_disease_proba(df, gamma)
#sns.kdeplot(data=df, x='disease_proba')
print(df[df.exposed==0].disease_proba.mean()*(df.exposed==0).sum())
print(df[df.exposed==1].disease_proba.mean()*(df.exposed==1).sum())
print(df[df.exposed==1].disease_proba.mean()/df[df.exposed==0].disease_proba.mean())
#%%
def simulate_disease(df, random_state=0):
    df = compute_disease_proba(df, gamma)
    if type(random_state)==int:
        rng = np.random.RandomState(random_state+2)
    else:
        rng = np.random
    df['disease'] = rng.rand(len(df))<df.disease_proba
    return df
dfd = simulate_disease(df)
ctd = matching.get_contingency_table(dfd)
matching.plot_heatmap(ctd)
OR, CI, pval = matching.compute_OR_CI_pval(
    dfd, print_=True, start_string='Estimated from whole population')
#%%
# first simulation
def get_true_OR(population_size):
    df = matching.simulate_exposure(beta, 3,0,
    population_size, random_state=False)
    df = simulate_disease(df)
    return matching.compute_OR_CI_pval(df) 
OR_ls, CI_ls, pval_ls =[],[],[]
population_size=2000

start = time.time()
#with mp.Pool(10) as pool:
#or_ci_pval_ls = pool.map(get_true_OR, 
#            [population_size for i in range(10)])
print(time.time()-start)
#%%    

fig, ax = plt.subplots(1,2)
_ = ax[0].hist(OR_ls, 30)
ax[0].set_xlabel('OR')
_ = ax[1].hist(pval_ls, 30)
ax[1].set_xlabel('p')
#%%

ctd = matching.get_contingency_table(dfd)
matching.plot_heatmap(ctd)
OR, CI, pval = matching.compute_OR_CI_pval(
    dfd, print_=True, start_string='Estimated from whole population')


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