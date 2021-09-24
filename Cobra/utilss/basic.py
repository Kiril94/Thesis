# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 10:41:26 2021

@author: klein
"""
import numpy as np


def create_dictionary(keys, values):
    result = {} # empty dictionary
    for key, value in zip(keys, values):
        result[key] = value
    return result

class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
def get_index(list_of_strings, substring):
    """search list of strings for substring and return index"""
    try:
        return next(i for i, e in enumerate(list_of_strings) if substring in e)
    except StopIteration:
        return len(list_of_strings) - 1
    
def my_argmax(a, axis=1, default=-1):
    rows = np.where(a == a.max(axis=1)[:, None])[0]
    rows_multiple_max = rows[:-1][rows[:-1] == rows[1:]]
    my_argmax = a.argmax(axis=1)
    my_argmax[rows_multiple_max] = default
    return my_argmax