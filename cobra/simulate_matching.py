import matplotlib
#%%
import numpy as np
import matplotlib.pyplot as plt
##%
def get_patients():
    pass
num_population = int(1e3)

def simulate_ps(b, num_population):
    x = np.random.rand(num_population)*10-5
    p = 1/(1+np.exp(-(b[0]+b[1]*x)))
    return x, p

def simulate_exposure(p):
    exposures = np.zeros(len(p))
    rand_arr = np.random.rand(len(p))
    exposures[p>rand_arr] = 1
    return exposures

def estimate_ps()
#def simulate_effect(exposures):


num_exposed = 1e2
num_unexposed = 1e4
#patients = get_patients()
b = np.array([1, 10])

x,p = simulate_ps(b, 1000)
exp = simulate_exposure(p)
fig, ax = plt.subplots()
ax.scatter(x, exp, color='r', s=2)
ax.scatter(x, p, s=.1)

#print(simulate_exposure)
