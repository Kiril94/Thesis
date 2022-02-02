#%%
import numpy as np
import matplotlib.pyplot as plt

from utilities import matching as mat
from utilities import matching
import seaborn as sns
import importlib
import pandas as pd
from scipy.stats import norm
importlib.reload(matching)
pd.options.mode.chained_assignment = None
#%% 
# Test


#%%
# Specify params
num_variables = 3
population_size = 1000
num_hidden_variables = 1
random_state = 0

#%%
# Simulate exposure
rng = np.random.RandomState(0)
loc = list(rng.rand(num_variables)*.5)
loc.insert(0, 4)
beta = norm.rvs(loc=loc, scale=1,size=num_variables+1, 
            random_state=random_state)
df = matching.simulate_exposure(beta, num_variables, population_size,
            random_state=random_state)
print('Number exposed', df.exposed.sum()

#%%
# Simulate disease

#%%
beta = np.array([.5, 10])
x, p = mat.simulate_ps_1var(Theta, 1000)
df = pd.DataFrame(x, columns=['x'])
exposure = mat.simulate_exposure(p)
df['exposed'] = exposure
p_exp = .1#probability of having the disease when bein exposed
true_OR = 10
p_nexp = p_exp/true_OR
def simulate_disease(df, p_exp, p_nexp):
    df_exp = df[df.exposed==1]
    df_exp['disease'] = np.random.rand(len(df_exp))<p_exp
    df_nexp = df[df.exposed==0]
    df_nexp['disease'] = np.random.rand(len(df_nexp))<p_nexp
    df = pd.concat([df_exp, df_nexp])
    return df
df = simulate_disease(df, p_exp, p_nexp)
fig, ax = plt.subplots()
_ = ax.scatter(x, exposure, color='r', s=2, label='True exposure')
_ = ax.scatter(x, p, s=.1, label='Probability of exposure')
ax.legend()
#%%
# Plot logistic regression
print(len(x))
print(len(exposure))
LR = matching.estimate_PS(x.reshape(-1,1), exposure)
fig, ax = plt.subplots(1,2)
e_PS = LR.predict_proba(x.reshape(-1,1))[:,1]
df['PS_estimate'] = e_PS
cases_mask = exposure==1
e_PS_cases = e_PS[cases_mask]
e_PS_controls = e_PS[~cases_mask]
x_cases = x[cases_mask]
x_controls = x[~cases_mask]
_ = ax[0].hist(e_PS_cases, label='cases', alpha=.4)
_ = ax[0].hist(e_PS_controls, label='controls', alpha=.4)
ax[0].set_xlabel('PS')
ax[0].set_ylabel('Count')
ax[0].legend()
_ = ax[1].scatter(x, exposure, color='r', s=2, label='True exposure')
x_ls = np.linspace(0,1,100)
ax[1].plot(x_ls, LR.predict_proba(x_ls.reshape(-1,1))[:,1],
    label= 'PS fit')
ax[1].plot(x_ls, matching.compute_PS(Theta, x_ls),
    label= 'true PS')
ax[1].set_xlabel('x')
ax[1].set_ylabel('PS')
ax[1].legend(loc=5, fontsize=10)
fig.tight_layout()
print(np.sum(exposure))
#%%
# Now that we have the PS, we can match patients


distances, indices = matching.NN_matching(
    e_PS_cases, e_PS_controls)
e_PS_matched_controls = e_PS_controls[indices.flatten()]
x_matched_controls = x_controls[indices.flatten()]

print('#cases', len(e_PS_cases))
print('#matched controls', len(e_PS_matched_controls))
fig, ax = plt.subplots(1,2)
_ = ax[0].hist(e_PS_cases, label='cases', alpha=.4)
_ = ax[0].hist(e_PS_matched_controls, label='controls', alpha=.4)
ax[0].set_xlabel('PS')
sns.kdeplot(x_cases, ax=ax[1], label='cases')
sns.kdeplot(x_matched_controls, ax=ax[1], label='matched controls')
ax[1].set_xlabel('x')  
#nn_OR = matching.compute_OR_p_value() 

#%%
def simulate_disease(df, p_exp, OR):
    df_exp = df[df.exposed==1]
    df_nexp = df[df.exposed==0]
    df_exp['disease'] = np.random.rand(len(df_exp))<p_exp
    df_nexp['disease'] = np.random.rand(len(df_nexp))<p_exp/OR
    df = pd.concat([df_exp, df_nexp])
    df['disease'] = df['disease']*1
    return df
# disease|no disease
#--------------------
#
def get_contingency_table(df):
    ed = (df.exposed==1) & (df.disease==1)
    end = (df.exposed==1) & (df.disease==0)
    ned = (df.exposed==0) & (df.disease==1)
    nend = (df.exposed==0) & (df.disease==0)
    #df = {[[ed, ned],[]]}
    return ed, end, ned, nend
    
df = simulate_disease(df, p_exp, true_OR)
def get_contingency_table(df):
    return pd.crosstab(index=df['exposed'], columns=df['disease'], margins=True)
ct = get_contingency_table(df)
print(ct)
#matching.compute_OR_p_value(ct)
#%%
df.head()
df.disease.hist()