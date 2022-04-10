import sys, os
from os.path import split, join
base_dir = os.getcwd()
print(base_dir)
if base_dir not in sys.path:
    sys.path.append(base_dir)
from os.path import join
from utilities import mri_stats
import pickle
from datetime import datetime as dt

fig_dir = join(base_dir, 'figs')
table_dir = join(base_dir, 'data/tables')
with open(join(table_dir, 'scan_tables','scan_after_sq_pred_dst_nos_date.pkl'), 'rb') as f:
    dfc = pickle.load(f)
dfc['positive_scan'] = 0
dfc.loc[dfc.days_since_test>=-3, 'positive_scan'] = 1
dfcc = dfc[(dfc['2019']==1)|(~dfc.InstanceCreationDate.isna())]
dfccc = dfcc[~dfcc.InstanceCreationDate.isna() & ~dfcc.InstanceCreationTime.isna()]
dfccc.loc[dfccc.index,'DateTime'] = dfccc.apply(lambda x: 
dt(x.InstanceCreationDate.year, 
    x.InstanceCreationDate.month,
    x.InstanceCreationDate.day,
    int(str(x.InstanceCreationTime)[:2]),
    int(str(x.InstanceCreationTime)[3:5]),
    0), axis=1
     )
df_sorted = dfccc.groupby('PatientID').apply(
        lambda x: (x.sort_values(by=['DateTime'], ascending=True)))
df_studies = mri_stats.get_studies_df(df_sorted, threshold=1)
df_studies.to_csv(f"{table_dir}/studies_clean.csv", index=False, header=True)