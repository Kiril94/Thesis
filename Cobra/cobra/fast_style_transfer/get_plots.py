"""
created on 9th may 2022
"""

#%%
from pathlib import Path
import numpy as np
import pandas as pd 
import json 
import matplotlib.pyplot as plt

file = open(Path(__file__).parent/"train.out")
lines = np.array(file.readlines()[3:])

odd_lines = lines[::2]
even_lines = lines[1::2]

columns=["Epoch","Iteration","Step","loss","style","content","tv"]
values = []                  
for (even_line,odd_line) in zip(odd_lines,even_lines):
    even_values = even_line.strip().split(", ")
    odd_values = odd_line.strip().split(", ")

    epoch = int(even_values[0][6:])
    iteration = int(even_values[1][11:])
    loss = float(even_values[2][6:])
    step = (epoch)*20000 + iteration
    
    style = float(odd_values[0][7:])
    content = float(odd_values[1][8:])
    tv = float(odd_values[2][4:])
    
    values.append([epoch,iteration,step,loss,style,content,tv])

#%%
df = pd.DataFrame(values,columns=columns)
df.to_csv("style_transfer_history.csv",index=False)    

fig,ax = plt.subplots(2,2,figsize=(10,7))
ax = ax.flatten()

ax[0].plot(df['Step'],df['loss'],label="loss",color="blue")
ax[0].plot(df['Step'],df['style'],label="style",color="orange")
ax[0].plot(df['Step'],df['content'],label="content",color="green")
ax[0].set(ylim=(-1000,5e4))

ax[1].plot(df['Step'],df['style'],label="style",color="orange")
ax[1].set(ylim=(-10,500))
ax[2].plot(df['Step'],df['content'],label="content",color="green")
ax[2].set(ylim=(-1000,4e4))
ax[3].plot(df['Step'],df['tv'],label="tv",color="red")
ax[3].set(ylim=(-300,1e4))

ax[0].legend()
ax[1].legend()
ax[2].legend()
ax[3].legend()

fig.suptitle("Syle transfer training")
fig.tight_layout()


