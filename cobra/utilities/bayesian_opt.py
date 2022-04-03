import xgboost as xgb
from skopt import BayesSearchCV
#creating deepcopy of model instances
from functools import wraps
from time import time
from skopt.space import Real,Integer
import json
import os
from os.path import join
cobra_dir = os.path.abspath('')
def timeit(func):
    @wraps(func)
    def _time_it(*args, **kwargs):
        start = int(round(time() * 1000))
        try:
            return func(*args, **kwargs)
        finally:
            end_ = int(round(time() * 1000)) - start
            if (end_>1000) and (end_<=60*1000):
                print(f"Total execution time: {round(end_/1000,2)} s")    
            elif (end_>60*1000) and (end_<=60*1000*60):
                print(f"Total execution time: {round(end_/1000/60,2)} min")    
            elif end_>60*1000*60:
                print(f"Total execution time: {round(end_/1000/60/60,2)} min")    
            else:
                print(f"Total execution time: {end_ if end_ > 0 else 0} ms")
    return _time_it



@timeit
def get_default_params(X_train, y_train, objective='multi:softprob',
                          eval_metric='mlogloss',
                          use_label_encoder=False):
    print("Obtain default parameters")
    #obtaining default parameters by calling .fit() to XGBoost model instance
    xgbc0 = xgb.XGBClassifier(objective=objective,
                          tree_method='auto',
                          eval_metric=eval_metric,
                          use_label_encoder=False)
    xgbc0.fit(X_train , y_train)

    #extracting default parameters from benchmark model
    default_params = {}
    gparams = xgbc0.get_params()
    #default parameters have to be wrapped in lists - even single values - so GridSearchCV can take them as inputs
    for key in gparams.keys():
        gp = gparams[key]
        default_params[key] = [gp]
    return default_params

#creating deepcopy of default parameters before manipulations

@timeit
def find_best_params(X_train, y_train, nfold=5,n_iter=100,
    param_grid= {
        'eta':Real(0.01, 0.3, 'log-uniform'),
        'gamma': Real(0,100, 'log-uniform'),
        'max_depth': Integer(3,10),
        'learning_rate': Real(0.01, 0.7),
        'min_child_weight':Integer(1,10),
        'subsample':Real(.5,1),
        'colsample_bytree':Real(.5,1),
        'lambda':Real(0,5),
        'alpha':Real(0,5),
            }):
    """https://towardsdatascience.com/binary-classification-xgboost-hyperparameter-tuning-scenarios-by-non-exhaustive-grid-search-and-c261f4ce098d
    typical hyperparams: 
    https://www.kaggle.com/code/prashant111/a-guide-on-xgboost-hyperparameters-tuning/notebook"""
    
    default_params = get_default_params(X_train, y_train)
    with open(join(cobra_dir, 'xgboost', 'default_params.txt'), 'w') as f:
        json.dump(default_params, f)
    #unwrapping list values of default parameters
    default_params_xgb = {}
    #No. of jobs
    
    for key in default_params.keys():
        default_params_xgb[key] = default_params[key][0]
    xgbc = xgb.XGBClassifier(**default_params_xgb)
    clf = BayesSearchCV(
        estimator=xgbc, search_spaces=param_grid, n_iter=n_iter, 
        scoring='accuracy', cv=nfold, return_train_score=True, verbose=3)
    clf.fit(X_train, y_train)
    #best parameters
    print('Finished')
    return clf.best_params_