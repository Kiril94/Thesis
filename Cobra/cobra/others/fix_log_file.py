
from pathlib import Path
import pandas as pd 

file_path = Path(__file__).parents[1] / 'tables' / 'log_to_download_swi.txt'

file = open(file_path,'r')
contents = file.read()[9:]

ids_len = 32
ids = [ contents[i:i+ids_len] for i in range(0,len(contents),ids_len) ]

df = pd.DataFrame({'PatientID':ids})
df.to_csv(Path(__file__).parents[1] / 'tables' / 'log_to_download_swi_post.txt',index=False)