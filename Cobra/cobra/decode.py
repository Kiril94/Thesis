#%%
import pandas as pd
from cryptography.fernet import Fernet
from pathlib import PurePath as Path
import os
from os.path import join
import ast

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
script_dir = os.path.realpath(__file__)
base_dir = Path(script_dir).parent
key = "1b3KCzziTwLPiqneoY8XMEQ2DhWpxixIeiRhLIWwZe4="
df_DST = pd.read_csv(join(base_dir,'data/share/sp/import/dst.csv'),
    converters={'DST':_parse_bytes}) 
fernet = Fernet(key)
df_DST.DST = df_DST.DST.map(lambda x: bytes(x, encoding='utf8'))
df_DST.DST = df_DST.DST.map(lambda x: fernet.decrypt(x).decode())
df_DST.DST = df_DST.DST.map(lambda x: int(x))
#%%
df_DST.nunique()