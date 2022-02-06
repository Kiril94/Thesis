import numpy as np
from scipy.stats import fisher_exact, norm
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.random import default_rng

def get_rand_uniform(num_variables, random_state=0):
    if type(random_state)==int:
        rng = np.random.RandomState(random_state)
        return rng.rand(num_variables)
    else:
        return np.random.rand(num_variables)

def compute_PS(beta, X):
    return 1/(1+np.exp(-(beta[0]+beta[1:]@X.T)))

def compute_disease_proba(df, gamma):
    variables_cols = [col for col in df.keys() \
        if col.startswith('x') or col.startswith('hx')]
    disease_proba =  1/(1+np.exp(gamma[0]+gamma[1]*df.exposed\
        +gamma[2:]@df[variables_cols].T))
    df['disease_proba'] = disease_proba
    return df

def simulate_variables_and_PS(beta, num_variables, population_size, random_state=0):
    X = norm.rvs(size=(population_size, num_variables), 
                random_state=random_state)
    true_PS = compute_PS(beta, X)
    return X, true_PS 

def simulate_exposure(beta, num_hidden_variables,
        population_size, random_state=0):
    """Returns dataframe with simulated, gaussian variables (mu=0, sig=1) and 
    exposure (0 or 1) for population_size patients. 
    Exposure is simulated using probabilities from a logistic model 
    (true propensity score)"""
    num_variables = len(beta)-1
    X, true_PS = simulate_variables_and_PS(
        beta, num_variables, population_size, random_state=random_state)
    exposures = np.zeros(len(X))
    if type(random_state)==int:
        rng = np.random.RandomState(random_state+1)
        exposures[true_PS>rng.rand(len(X))] = 1
    else:
        exposures[true_PS>np.random.rand(len(X))] = 1
    df_data = np.concatenate([X, exposures.reshape(-1,1)], axis=1)
    df_columns = ['x'+str(i) for i in range(num_variables-num_hidden_variables)]
    df_columns = df_columns + ['hx'+str(i) for i in range(num_hidden_variables)]
    df_columns.append('exposed')
    df = pd.DataFrame(data=df_data, columns=df_columns)
    return df

def simulate_disease(df, odds_exp, OR, random_state=0):
    """
    Parameters:
        df: Df with exposed column (0 or 1)
        odds_exp: odds of getting the disease when being exposed
        OR: odds ratio
    returns: Dataframe with disease column (0 or 1)"""
    df_exp = df[df.exposed==1]
    df_nexp = df[df.exposed==0]
    
    if type(random_state)==int:
        rng1 = np.random.RandomState(random_state+2)
        rng2 = np.random.RandomState(random_state+3)
    else:
        rng1 = np.random
        rng2 = np.random
    df_exp['disease'] = rng1.rand(len(df_exp))<odds_exp
    df_nexp['disease'] = rng2.rand(len(df_nexp))<odds_exp/OR
    df = pd.concat([df_exp, df_nexp], ignore_index=True)
    df['disease'] = df['disease']*1
    return df


# evaluation
def get_contingency_table(df, two_by_two=True):
    ct = pd.crosstab(index=df['exposed'], columns=df['disease'], margins=True)
    if two_by_two:
        return np.array(ct.iloc[:2,:2])
    else:
        return ct

def plot_heatmap(ct):
    denomenator = np.repeat(np.sum(ct, axis=1).reshape(-1,1), 2, axis=1)
    fig, ax = plt.subplots(1,2)
    sns.heatmap(ct, annot=True, fmt="d", ax=ax[0])
    sns.heatmap(ct/denomenator, 
        annot=True, fmt=".2f", ax=ax[1])
    ax[0].set_ylabel('exposure')
    ax[1].set_ylabel('exposure')
    ax[0].set_xlabel('disease')
    ax[1].set_xlabel('disease')
    fig.tight_layout()

def get_num_variables(df):
    return len([k for k in df.keys() if k.startswith('x')])
def get_num_hidden_variables(df):
    return len([k for k in df.keys() if k.startswith('hx')])

def plot_variables_kde(df, hue='exposed'):
    num_variables = get_num_variables(df)
    fig, ax = plt.subplots(1,num_variables)
    if num_variables!=1:
        ax = ax.flatten()
        for i, a in enumerate(ax):
            if i<num_variables:
                sns.kdeplot(data=df, x='x'+str(i), hue=hue, ax=a,)
    else:
        sns.kdeplot(data=df, x='x0', hue=hue, ax=ax,)
    fig.tight_layout()


def compute_OR_pval(df):
    ct = get_contingency_table(df)
    OR, pval = fisher_exact(ct) 
    return OR, pval
    
def compute_logOR_SE(df):
    logOR = np.log(compute_OR_pval(df)[0])
    ct = get_contingency_table(df)
    SE = np.sqrt(np.sum(1/ct))
    return logOR, SE 

def compute_OR_95CI(df):
    OR = compute_OR_pval(df)[0]
    ct = get_contingency_table(df)
    range_ = 1.96*np.sqrt(np.sum(1/ct))
    a = np.log(OR)-range_
    b = np.log(OR)+range_
    return (np.exp(a), np.exp(b))

def compute_OR_CI_pval(df, print_=False, start_string=''):
    ct = get_contingency_table(df)
    CI = compute_OR_95CI(df)
    OR, p_val = fisher_exact(ct) 
    if print_:
        print(start_string, "\n     OR (95% CI) =", OR, 
            f'({CI[0]:.2f},{CI[1]:.2f})', '\n    p =', p_val)
    return OR, CI, p_val


def estimate_PS(df, random_state=0):
    num_variables = get_num_variables(df)
    variables_columns = ['x'+str(i) for i \
        in range(num_variables)]
    X = df[variables_columns]
    y = df['exposed']
    if type(random_state)==int:
        LR = LogisticRegression(random_state=random_state).fit(X, y)
    else:
        LR = LogisticRegression().fit(X, y)
    df['PS'] = LR.predict_proba(X)[:,1]
    return df


def get_positives_and_random_subset(df, n_subset, random_state=0):
    """Selects all the sick and a random subset of size n_subset of the rest of the population"""
    dfd = df[df.disease==1]
    dfnd = df[df.disease==0]
    deck = np.arange(n_subset)
    if type(random_state)==int:
        rng = default_rng(random_state)
        rng.shuffle(deck)
    else:
        deck = np.random.shuffle(deck)
    df_rand = dfnd.iloc[deck, :].reset_index()
    df_subs = pd.concat([dfd, df_rand], ignore_index=True)
    return df_subs


# actual matching

def NN_matching(df):
    PS_cases = df[df.disease==1].PS.toarray()
    PS_controls = df[df.disease==0].PS.toarray()
    neigh = NearestNeighbors(n_neighbors=10)
    neigh.fit(PS_controls)
    distances, indices = neigh.kneighbors(PS_cases)
    return distances, indices

