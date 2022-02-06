#%%
# Import
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utilities import matching
import importlib
from numpy.random import default_rng

from scipy.stats import norm
importlib.reload(matching)
import multiprocessing as mp
pd.options.mode.chained_assignment = None

#%%
# Specify params
num_variables = 1
population_size = 15000
num_hidden_variables = 0
true_OR = 5
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
#def get_gamma1(OR):
#    """Calculate gamma in gamma*t from the OR (rare disease assumption)"""
#    return np.log(OR)
def get_gamma1(true_OR, gamma0):
    """Calculate gamma in gamma*t from the OR (rare disease assumption)"""
    np.log(true_OR)-np.log(1-(true_OR-1)*np.exp(gamma0))
    return np.log(true_OR)
def crude_estimation_OR(df):
    return df[df.exposed==1].disease_proba.mean()/df[df.exposed==0].disease_proba.mean()
def crude_estimation_dis1(df):
    return df[df.exposed==1].disease_proba.mean()*(df.exposed==1).sum()
def crude_estimation_dis0(df):
    return df[df.exposed==0].disease_proba.mean()*(df.exposed==0).sum()
print("gamma0<=", np.log(1/(true_OR-1)))
gamma0 = -6
gamma1 = get_gamma1(true_OR, gamma0)
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
print("exposed and sick: ", crude_estimation_dis1(df))
print("exposed and not sick: ", crude_estimation_dis0(df))
print("estimation of OR:" ,crude_estimation_OR(df))
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
OR_all, CI_all, pval_all = matching.compute_OR_CI_pval(
    dfd, print_=True, start_string='Estimated from whole population')
logOR_all, logORSE_all = matching.compute_logOR_SE(dfd)
print(logOR_all, logORSE_all)
#%%
matching.plot_variables_kde(dfd)
matching.plot_variables_kde(dfd, hue='disease')
#%%
# Select random subset
def get_positives_and_random_subset(df, n_subset, random_state=0):
    """Selects all the sick and a random subset of size n_subset of the rest of the population"""
    dfd = df[df.disease==1]
    dfnd = df[df.disease==0]
    deck = np.arange(len(dfnd))
    if type(random_state)==int:
        rng = default_rng(random_state)
        rng.shuffle(deck)
    else:
        np.random.shuffle(deck)
    df_rand = dfnd.iloc[deck[:n_subset], :].reset_index()
    df_subs = pd.concat([dfd, df_rand], ignore_index=True)
    return df_subs
def check_OR(OR_all, CI_nm):
    if CI_nm[0]<OR_all and CI_nm[1]>OR_all:
        return 1
    else:
        return 0
def z_test(mu1, mu2, sig1, sig2):
    return np.abs(mu1-mu2)/np.sqrt(sig1**2+sig2**2)

OR_ls = []
OR_within_ls = []
z_val_ls = []
num_disease = len(dfd[dfd.disease==1])
for i in range(40):
    dfs = get_positives_and_random_subset(dfd, 500-num_disease, random_state=None)
    OR_nm, CI_nm, pval_nm = matching.compute_OR_CI_pval(dfs, print_=False, 
        start_string='No matching')
    logOR_subs, logORSE_subs = matching.compute_logOR_SE(dfs)
    z_val_ls.append(z_test(logOR_all, logORSE_subs, logORSE_all, logORSE_subs))
    OR_within_ls.append(check_OR(OR_all, CI_nm))
    OR_ls.append(OR_nm)
print(sum(OR_within_ls))
fig, ax = plt.subplots(1,2)
_ = ax[0].hist(OR_ls, 30)
_ = ax[1].hist(z_val_ls, 30)
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