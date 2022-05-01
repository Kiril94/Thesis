
import numpy as np
import xgboost as xgb
from typing import Tuple

def gradient(predt: np.ndarray, dtrain: xgb.DMatrix, delta) -> np.ndarray:
    '''Compute the gradient huber.'''
    y = dtrain.get_label()
    d = predt-y
    out = np.zeros(d.shape)
    out[np.abs(d)<=delta] = d
    out[d>delta] = delta
    out[d<delta] = -delta
    return out

def hessian(predt: np.ndarray, dtrain: xgb.DMatrix, delta) -> np.ndarray:
    '''Compute the hessian for squared log error.'''
    y = dtrain.get_label()
    d = predt-y
    out = np.zeros(d.shape)
    out[np.abs(d)<=delta] = 1
    return out
    
def huber(predt: np.ndarray,
                dtrain: xgb.DMatrix, delta=1) -> Tuple[np.ndarray, np.ndarray]:
    '''Huber objective.
    '''
    grad = gradient(predt, dtrain, delta)
    hess = hessian(predt, dtrain, delta)
    return grad, hess