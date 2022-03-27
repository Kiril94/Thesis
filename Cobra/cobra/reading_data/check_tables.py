import pandas as pd 
from pathlib import Path

fp = Path(__file__)
swi_all = pd.read_csv(fp.parent.parent/"tables"/"swi_all.csv")
negposclean = pd.read_csv(fp.parent.parent/"tables"/"neg_pos_clean.csv")

swi_from_neg_posclean = negposclean[ negposclean['Sequence']=='swi' ] 

print(swi_from_neg_posclean.shape)
print(swi_all.shape)

swi_all_notin_negposclean = swi_all[ ~swi_all['SeriesInstanceUID'].isin(swi_from_neg_posclean['SeriesInstanceUID'])]
print(swi_all_notin_negposclean)