#%%
import pandas as pd
import numpy as np
#%%
# In[Parahippocampal gyrus]
dfm = pd.read_csv('DK_template.csv')
all_keys = list(dfm.keys())
all_keys.remove('Left-parahippocampal')
all_keys.remove('Right-parahippocampal')
all_keys.remove('Image-name-unique')
dfm[all_keys] = 1
dfm['Left-parahippocampal'] = 0
dfm['Right-parahippocampal'] = 0

dfm.iloc[[0]].to_csv('DK_parahipp.csv', index=None)

#%%
# In[Parahippocampal gyrus]
dfm = pd.read_csv('DK_template.csv')
all_keys = list(dfm.keys())
all_keys.remove('Left-temporalpole')
all_keys.remove('Right-temporalpole')
all_keys.remove('Image-name-unique')
dfm[all_keys] = 1
dfm['Left-temporalpole'] = 0
dfm['Right-temporalpole'] = 0
dfm.iloc[[0]].to_csv('DK_temp_pole.csv', index=None)

#%%
# In[Anterior Cingulate Cortex]
dfm = pd.read_csv('DK_template.csv')
all_keys = list(dfm.keys())
keys_highlight = ['Left-rostralanteriorcingulate',
                'Right-rostralanteriorcingulate',
                'Left-caudalanteriorcingulate',
                'Right-caudalanteriorcingulate'] 
for k in keys_highlight:
    all_keys.remove(k)
all_keys.remove('Image-name-unique')
dfm[all_keys] = 1
for k in keys_highlight:
    dfm[k] = 0
dfm.iloc[[0]].to_csv('DK_anterior_cingulate.csv', index=None)

#%%
#In[Rainbow]
dfm = pd.read_csv('DK_template.csv')
dfm.iloc[0,1:] = np.random.uniform(size=len(dfm.keys())-1)*5
dfm.iloc[[0]].to_csv('DK_rainbow.csv', index=None)
