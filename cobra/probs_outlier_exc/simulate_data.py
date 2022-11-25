#%%
import numpy as np
import matplotlib.pyplot as plt
def get_mu(ages):
    return np.sigmoid(ages - 50)
#%%
fig, ax = plt.subplots()
ages = np.arange(0, 100)
ax.plot(ages, get_mu(ages))
ax.set_xlabel('Age')   
ax.set_ylabel('mu')
plt.show()