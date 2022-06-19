#%%
import pandas as pd
import numpy as np
import pickle
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


#%%
#In[z-values]
# Load dict
def get_key_ls(keys, st):
    return [k for k in keys if st in k]
with open('zvalues_cov.pkl', 'rb') as f:
    zdc = pickle.load(f)
#print(zdc)
print(zdc)
dfm = pd.read_csv('DK_template.csv')
keys = list(dfm.keys()) 
br_dic = {'Rolandic Operculum':get_key_ls(keys, 'parsopercularis')+\
        get_key_ls(keys, 'triangularis')+get_key_ls(keys, 'parsorbitalis')+\
        get_key_ls(keys, 'superiortemporal'), 
        'Orbitofrontal Cortex':get_key_ls(keys,'orbitofrontal'),
        'Heschls Gyrus':get_key_ls(keys, 'transverse'),
        'Cingulate Gyrus':[],
        'Bilateral Insulas':[],
        'Bilateral Hipocampi':get_key_ls(keys, 'Hippo'),
        'Parahippocampal Gyrus':get_key_ls(keys, 'parahi'),
        'Temporal Pole':get_key_ls(keys, 'temporalpole'),
        'Anterior Cingulate Cortex':[],
        'Supramarginal Gyrus':get_key_ls(keys, 'supramarginal'),
        'Ventricles':get_key_ls(keys, 'Ventricle')+get_key_ls(keys, 'Lat-Vent'),
        'Whole Brain':[]
        }
print(br_dic)
#'Anterior Cingulate Cortex':get_key_ls(keys, 'anteriorcingulate'),
#'Bilateral Insulas':get_key_ls(keys,'insula'),
# Set z-values
keys.remove('Image-name-unique')
dfm[keys] = 0
for br, z in zdc.items():
    for br_temp in br_dic[br]:
        dfm[br_temp] = z
#dfm.iloc[[0]].to_csv('DK_z_values_cov.csv', index=None)
dfm.head()

#%%
#In[plot all brain regions involved]
# Load dict
def get_key_ls(keys, st):
    return [k for k in keys if st in k]

dfm = pd.read_csv('DK_template.csv')
keys = list(dfm.keys()) 
br_dic = {'Rolandic Operculum':get_key_ls(keys, 'parsopercularis')+\
        get_key_ls(keys, 'triangularis')+get_key_ls(keys, 'parsorbitalis')+\
        get_key_ls(keys, 'superiortemporal'), 
        'Orbitofrontal Cortex':get_key_ls(keys,'orbitofrontal'),
        'Heschls Gyrus':get_key_ls(keys, 'transverse'),
        'Cingulate Gyrus':[],
        'Bilateral Insulas':[],
        'Bilateral Hipocampi':get_key_ls(keys, 'Hippo'),
        'Parahippocampal Gyrus':get_key_ls(keys, 'parahi'),
        'Temporal Pole':get_key_ls(keys, 'temporalpole'),
        'Anterior Cingulate Cortex':[],
        'Supramarginal Gyrus':get_key_ls(keys, 'supramarginal'),
        'Ventricles':get_key_ls(keys, 'Ventricle')+get_key_ls(keys, 'Lat-Vent'),
        'Whole Brain':[]
        }
print(br_dic)
#'Anterior Cingulate Cortex':get_key_ls(keys, 'anteriorcingulate'),
#'Bilateral Insulas':get_key_ls(keys,'insula'),
# Set z-values
keys.remove('Image-name-unique')
dfm[keys] = 0
for br in br_dic.keys():
    for br_temp in br_dic[br]:
        print(br_temp)
        dfm[br_temp] = 1
dfm.iloc[[0]].to_csv('defense_all_brs.csv', index=None)
dfm.head()

# %%
#In[Visualize all included brain regions]
# Load dict
def get_key_ls(keys, st):
    return [k for k in keys if st in k]


dfm = pd.read_csv('DK_template.csv')
keys = list(dfm.keys()) 

        
#print(br_dic)
#'Anterior Cingulate Cortex':get_key_ls(keys, 'anteriorcingulate'),
#'Bilateral Insulas':get_key_ls(keys,'insula'),
# Set z-values
br_dic1 =  {'Rolandic Operculum':get_key_ls(keys, 'parsopercularis')+\
        get_key_ls(keys, 'triangularis')+get_key_ls(keys, 'parsorbitalis')+\
        get_key_ls(keys, 'superiortemporal'), 
        'Orbitofrontal Cortex':get_key_ls(keys,'orbitofrontal'),
        'Heschls Gyrus':get_key_ls(keys, 'transverse'),
        'Cingulate Gyrus':[],
        'Bilateral Insulas':[],}
br_dic2 = {'Bilateral Hipocampi':get_key_ls(keys, 'Hippo'),
        'Parahippocampal Gyrus':get_key_ls(keys, 'parahi'),
        'Temporal Pole':get_key_ls(keys, 'temporalpole'),
        'Anterior Cingulate Cortex':[],
        'Supramarginal Gyrus':get_key_ls(keys, 'supramarginal'),
        'Ventricles':get_key_ls(keys, 'Ventricle')+get_key_ls(keys, 'Lat-Vent'),
        }
nums = [np.arange(1,6)]
keys.remove('Image-name-unique')
dfm[keys] = 0
for num, br in zip(nums, br_dic1.keys()):
    for br_temp in br_dic1[br]:
        dfm[br_temp] = num
dfm.iloc[[0]].to_csv('all_brain_regions1.csv', index=None)
#dfm.head()

#%%
get_key_ls(keys, 'cing')
#print(dfm.keys())
#%%
#In[Color whole brain]
dfm = pd.read_csv('DK_template.csv')
keys = list(dfm.keys())
keys.remove('Image-name-unique')
dfm[keys] = zdc['Whole Brain']
dfm.iloc[[0]].to_csv('DK_z_values_cov_wb.csv', index=None)
#%%
# %%
import scipy.stats as ss
print(ss.norm.cdf(-zdc['Whole Brain']))