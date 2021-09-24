# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 13:57:01 2021

@author: klein
"""
import numpy as np
from . import basic


#######################Mutliclass classification####################################
def fp_multi(cm):
    """Number of false positives, Takes confusion matrix"""
    return cm.sum(axis=0) - np.diag(cm)  

def fn_multi(cm):
    """Number of false negatives, Takes confusion matrix"""
    return cm.sum(axis=1) - np.diag(cm)

def tp_multi(cm):
    """Number of true positives, Takes confusion matrix"""
    return np.diag(cm)
def tn_multi(cm):
    """Number of true negatives, Takes confusion matrix"""
    fp = fp_multi(cm)
    fn = fn_multi(cm)
    tp = tp_multi(cm)
    tn = cm.sum() - (fp + fn + tp)
    return tn

def tpr_multi(cm):
    """Takes confusion matrix
    returns: Sensitivity, hit rate, recall, or true positive rate"""
    return tp_multi(cm)/(tp_multi(cm)+fn_multi(cm))
def tnr_multi(cm):
    """Takes confusion matrix
    returns: Specificity or true negative rate"""  
    return tn_multi(cm)/(tn_multi(cm)+fp_multi(cm))

def ppv_multi(cm):
    """Takes confusion matrix
    returns: Precision or positive predictive value"""  
    return tp_multi(cm)/(tp_multi(cm)+fp_multi(cm))

def npv_multi(cm):
    """Takes confusion matrix
    returns: Negative predictive value"""  
    return tn_multi(cm)/(tn_multi(cm)+fn_multi(cm))

def fpr_multi(cm):
    """Takes confusion matrix
    returns: Fall out or false positive rate"""  
    return fp_multi(cm)/(fp_multi(cm)+tn_multi(cm))

def fnr_multi(cm):
    """Takes confusion matrix
    returns: False negative rate"""  
    return fn_multi(cm)/(tp_multi(cm)+fn_multi(cm))

def fdr_multi(cm):
    """Takes confusion matrix
    returns: False discovery rate"""  
    return fp_multi(cm)/(tp_multi(cm)+fp_multi(cm))

def acc_multi(cm):
    """Takes confusion matrix
    returns: Overall accuracy"""  
    return (tp_multi(cm)+tn_multi(cm))/(tp_multi(cm)+fp_multi(cm)+fn_multi(cm)+tn_multi(cm))

########prediction################
def prob_to_class(pred_prob, threshold, other_class):
    """Takes predicted probabilities of shape (n_samples, n_classes)
    and a threshold
    Returns: array of n_samples with predicted classes, class is selected if
    probability is above threshold, else other_class is selected."""
    y = np.copy(pred_prob)
    y[y>threshold] = 1
    y[y<=threshold] = 0
    y = basic.my_argmax(y, axis=1, default=other_class)
    return y