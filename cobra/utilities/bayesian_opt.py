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
def find_best_params(X_train, y_train, n_jobs=1,n_points=1,nfold=5, n_iter=100, 
    objective='multi:softprob',
    eval_metric='mlogloss',
    search_space= {
        'eta':Real(0.01, 0.3, 'log-uniform'),
        'gamma':Real(1e-9,1, 'log-uniform'),
        'max_depth': Integer(3,10),
        'learning_rate': Real(0.001, 0.7),
        'min_child_weight':Integer(1,10),
        'subsample':Real(.3,1),
        'colsample_bytree':Real(.3,1),
        'reg_lambda':Real(0,5),
        'reg_alpha':Real(0,5),
    }):
    
    """https://towardsdatascience.com/binary-classification-xgboost-hyperparameter-tuning-scenarios-by-non-exhaustive-grid-search-and-c261f4ce098d
    typical hyperparams: 
    https://www.kaggle.com/code/prashant111/a-guide-on-xgboost-hyperparameters-tuning/notebook"""
    xgbc = xgb.XGBClassifier(tree_method='auto',eval_metric=eval_metric,
        objective=objective, use_label_encoder=False)
    clf = BayesSearchCV(
        estimator=xgbc, search_spaces=search_space, n_iter=n_iter, 
        cv=nfold, return_train_score=True, verbose=3, n_jobs=n_jobs, n_points=n_points)
    clf.fit(X_train, y_train)
    #best parameters
    print('Finished')
    return clf.best_params_