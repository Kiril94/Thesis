# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 10:41:26 2021

@author: klein
"""
import numpy as np
import pandas as pd


def create_dict(keys, values):
    return dict(zip(keys, values))


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def inv_dict(dict_):
    """creates inverse of a dictionary."""
    return {v: k for k, v in dict_.items()}


def get_index(list_of_strings, substring):
    """search list of strings for substring and return index"""
    try:
        return next(i for i, e in enumerate(list_of_strings) if substring in e)
    except StopIteration:
        return len(list_of_strings) - 1


def my_argmax(a, axis=1, default=-1):
    """Same as np.argmax but if multiple max values are present
    the values is set to default and not to 0 as np does."""
    rows = np.where(a == a.max(axis=axis)[:, None])[0]
    rows_multiple_max = rows[:-1][rows[:-1] == rows[1:]]
    my_argmax = a.argmax(axis=axis)
    my_argmax[rows_multiple_max] = default
    return my_argmax


def df_to_dict(df, col_key, col_val):
    """Create a dict from a df using two columns"""
    return pd.Series(df[col_val].values, index=df[col_key]).to_dict()