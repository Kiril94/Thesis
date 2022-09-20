#%%
from statsmodels.stats.multitest import fdrcorrection
pvalues = [0.04, 0.13, 0.06, 0.25, 0.77,
        0.03,0.01,0.07,0.06,0.25,0.77,
        0.01,0.59,0.01,0.59,0.01,0.59,0.03,
        0.14,0.06,0.25,0.69,0.04
        ]
rejected, q_value = fdrcorrection(pvalues)
#%%
print(rejected, q_value)