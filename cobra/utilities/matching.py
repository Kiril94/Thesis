import numpy as np
from scipy.stats import fisher_exact, beta
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression

def compute_PS(b, x):
    return 1/(1+np.exp(-(b[1]*(x-b[0]))))


def simulate_ps_1var(b, num_population):
    x = beta.rvs(a=2, b=4, size=num_population)
    p = compute_PS(b, x)
    return x, p

def simulate_exposure(p):
    exposures = np.zeros(len(p))
    rand_arr = np.random.rand(len(p))
    exposures[p>rand_arr] = 1
    return exposures

def compute_OR_p_value(de, he, dne, hne):
    ct = np.array([[de,he],[dne,hne]])
    OR, p_val = fisher_exact(ct) 
    return OR, p_val

def compute_OR_95CI(de, he, dne, hne):
    OR = compute_OR_p_value(de, he, dne, hne)[0]
    range_ = 1.96*np.sqrt(1/de+1/he+1/dne+1/hne)
    a = np.log(OR)-range_
    b = np.log(OR)+range_
    return (np.exp(a), np.exp(b))

def estimate_PS(X, y):
    LR = LogisticRegression(random_state=42).fit(X, y)
    LR.predict(X) #Return the predictions
    return LR


def NN_matching(PS_cases, PS_controls):
    neigh = NearestNeighbors(n_neighbors=10)
    neigh.fit(PS_controls.reshape(-1,1))
    distances, indices = neigh.kneighbors(PS_cases.reshape(-1,1))
    return distances, indices