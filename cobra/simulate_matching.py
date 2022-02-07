
#%%
# Import
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utilities import matching as mt
from stats_tools import vis
from stats_tools.vis import quickplot
import importlib
from numpy.random import default_rng
from scipy.stats import norm
importlib.reload(mt)
importlib.reload(vis)
import multiprocessing as mp
pd.options.mode.chained_assignment = None


#%%
# Specify params
num_variables = 1
population_size = 15000
num_hidden_variables = 0
true_OR = 1.5
random_state = 0
#%%
# Simulate exposure
#beta_loc = list(get_rand_uniform(num_variables)-.5)
#beta_loc.insert(0, -1)
#beta = norm.rvs(loc=beta_loc, scale=1, size=num_variables+1, 
#            random_state=2)
beta = np.array([-2,1])
df = mt.simulate_exposure(beta, num_hidden_variables,
     population_size, random_state=random_state)
quickplot(df.x0, y=df.exposed, xlabel='x0', ylabel='exposed')
print('Number exposed', df.exposed.sum())
quickplot(df.x0, xlabel='x0', ylabel='Count')
#%%
# test
#gamma_loc = list(get_rand_uniform(num_variables,2)*0.0001-2)
#gamma_loc.insert(0, gamma1)
#gamma_loc.insert(0, gamma0)
#gamma = norm.rvs(loc=gamma_loc, scale=.0001, 
#            size=num_variables+2, 
#            random_state=random_state+1)    

gamma = mt.get_gamma(gamma0=-4.5, true_OR=true_OR, gamma_ls=[.01])

df = mt.compute_disease_proba(df, gamma)
sns.kdeplot(data=df, x='disease_proba', hue='exposed')

print("exposed and sick: ", mt.crude_estimation_dis1(df))
print("not exposed and sick: ", mt.crude_estimation_dis0(df))
print("estimation of OR:" , mt.crude_estimation_OR(df))
#%%
def simulate_disease(df, random_state=0):
    df = mt.compute_disease_proba(df, gamma)
    if type(random_state)==int:
        rng = np.random.RandomState(random_state+2)
    else:
        rng = np.random
    df['disease'] = rng.rand(len(df))<df.disease_proba
    return df
dfd = simulate_disease(df)
ctd = mt.get_contingency_table(dfd)
mt.plot_heatmap(ctd)
OR_all, CI_all, pval_all = mt.compute_OR_CI_pval(
    dfd, print_=True, start_string='Estimated from whole population')
logOR_all, logORSE_all = mt.compute_logOR_SE(dfd)
print(logOR_all, logORSE_all)
#%%
mt.plot_variables_kde(dfd)
mt.plot_variables_kde(dfd, hue='disease')
#%%
# Select random subset



OR_ls = []
OR_within_ls = []
z_val_ls = []
num_disease = len(dfd[dfd.disease==1])
for i in range(5):
    dfs = mt.get_positives_and_random_subset(dfd, 7000-num_disease, random_state=None)
    OR_nm, CI_nm, pval_nm = mt.compute_OR_CI_pval(dfs, print_=False, 
        start_string='No mt')
    logOR_subs, logORSE_subs = mt.compute_logOR_SE(dfs)
    z_val_ls.append(z_test(logOR_all, logORSE_subs, logORSE_all, logORSE_subs))
    OR_within_ls.append(check_OR(OR_all, CI_nm))
    OR_ls.append(OR_nm)
print(sum(OR_within_ls))
fig, ax = plt.subplots(1,2)
_ = ax[0].hist(OR_ls, 30)
_ = ax[1].hist(z_val_ls, 30)
#%%
# Estimate PS
df_subs = mt.run_logistic_regression(df_subs)
sns.histplot(data=df_subs, x='PS', bins=50, hue='exposed')
df.head()
#%%
from sklearn.neighbors import NearestNeighbors
def NN_mt(df):
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

df_subs_m = NN_mt(
    df_subs)
OR_nm, CI_nm, pval_nm = mt.compute_OR_CI_pval(df_subs_m, print_=True, 
    start_string='No mt')
ct = mt.get_contingency_table(df_subs_m)
mt.plot_heatmap(ct)
#e_PS_matched_controls = e_PS_controls[indices.flatten()]
#x_matched_controls = x_controls[indices.flatten()]
#%%
#
np.log(20)