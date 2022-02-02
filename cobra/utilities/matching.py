import numpy as np
from scipy.stats import fisher_exact, norm
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
import pandas as pd


def compute_PS(beta, X):
    return 1/(1+np.exp(beta[0]+beta[1:]@X.T))

def simulate_var_and_ps(beta, num_variables, population_size, random_state=0):
    X = norm.rvs(size=(population_size, num_variables), random_state=random_state)
    true_PS = compute_PS(beta, X)
    return X, true_PS 

def simulate_exposure(beta, num_variables, population_size, random_state=0):
    X, true_PS = simulate_var_and_ps(beta, num_variables, population_size)
    exposures = np.zeros(len(X))
    if type(random_state)==int:
        rng = np.random.RandomState(random_state)
        exposures[true_PS>rng.rand(len(X))] = 1
    else:
        exposures[true_PS>np.random.rand(len(X))] = 1
    df_data = np.concatenate([X, exposures.reshape(-1,1)], axis=1)
    df_columns = ['x'+str(i) for i in range(num_variables)]
    df_columns.append('exposed')
    df = pd.DataFrame(data=df_data, columns=df_columns)
    return df

def compute_OR_p_value(ed, end, ned, nend):
    ct = np.array([[ed,end],[ned,nend]])
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