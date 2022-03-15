"""Used to decode encrypted information exported from computerome."""

#%%
import pandas as pd
from cryptography.fernet import Fernet
from pathlib import PurePath as Path
import os
from os.path import join
import ast
import numpy as np


#%%
def _parse_bytes(field):
    """ Convert string represented in Python byte-string literal b'' syntax into
        a decoded character string - otherwise return it unchanged.
    """
    result = field
    try:
        result = ast.literal_eval(field)
    finally:
        return result.decode() if isinstance(result, bytes) else field
#%%
# In[Load and decrypt df]
script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parent
key = "1b3KCzziTwLPiqneoY8XMEQ2DhWpxixIeiRhLIWwZe4="
df_DST = pd.read_csv(join(base_dir,'data/share/sp/import/dst2.csv'),
    converters={'DST':_parse_bytes})[['PID','SID','DST']] 
fernet = Fernet(key)
df_DST.DST = df_DST.DST.map(lambda x: bytes(x, encoding='utf8'))
df_DST.DST = df_DST.DST.map(lambda x: fernet.decrypt(x).decode())
df_DST.DST = df_DST.DST.map(lambda x: int(x))
df_DST = df_DST.rename(columns={'PID':'PatientID', 'SID':'SeriesInstanceUID','DST':'days_since_test'})

#%%
# In[Save new decrypted df]
df_DST.to_csv(join(base_dir, 'data/tables/days_since_test2.csv'), index=False)
#%%

# In[Load df clean]
df_clean = pd.read_csv(join(base_dir, 'data/tables/neg_pos_clean.csv'))
df_clean.keys()
#%%
df_clean_new = pd.merge(df_clean, 
                    df_DST[['PatientID','SeriesInstanceUID', 'days_since_test']], 
                    how='left', on=['PatientID','SeriesInstanceUID'] )  

#%%
df_clean_new = df_clean_new.drop(columns=['days_since_test_x'])
df_clean_new  = df_clean_new.rename(columns={'days_since_test_y':'days_since_test'})
#%%
df_clean_new
#%%
df_clean.to_csv(join(base_dir, 'data/tables/neg_pos_clean.csv'), index=False)

#%%
# In[Create patient groups]
def get_patients_post(df, seq, threshold_post=(-3)):
    df = df[df.Sequence==seq]
    return df[df.days_since_test>=threshold_post].PatientID.unique()
def get_patients_pre_post(df, seq, threshold_pre=(-30), threshold_post=(-3)):
    post_pat = get_patients_post(df, seq, threshold_post=threshold_post)
    df = df[df.PatientID.isin(post_pat)]
    df = df[df.Sequence==seq]
    return df[df.days_since_test<=threshold_pre].PatientID.unique()

t1_pre_post = get_patients_pre_post(df_clean_new, 't1')

t1_post = get_patients_post(df_clean_new, 't1')
len(t1_post)
#%%
np.savetxt(join(base_dir, 'data/patient_groups/t1_post.txt'),
                t1_post,delimiter=" ", fmt="%s" )
# There are some patient missing!!!