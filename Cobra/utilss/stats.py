# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 11:41:20 2021

@author: klein
"""
import numpy as np
import itertools


def only_first_true(a, b):
    """takes two binary arrays
    and returns True
    where only the el. of the first array is true
    if only the second or both are true returns false"""
    return a&np.logical_not(a&b)

def mask_sequence_type(df, str_, key='SeriesDescription'):
    """Checks all the values in df/groupby
    in the column key (SeriesDescription by default),
    if they contain the string str_. Returns a mask.
    """
    try:
        mask = df[key].str.contains(str_, na=False)
    except:
        mask = df.apply(lambda x: x[key].str.contains(str_, na=False))
    return mask

def check_tags(df, tags, key='SeriesDescription'):
    """calls mask_sequence type for a list of tags and combines
    the masks with or"""
    masks = []
    for tag in tags:
        masks.append(mask_sequence_type(df, tag, key))
    mask = masks[0]
    for i in range(1, len(masks)):
        mask = mask | masks[i]
    return mask

def group_small(dict_, threshold, keyword='other'):
    """Takes a dictionary and sums all values that are smaller than threshold
    the result is stored under the key keyword. Useful for pie charts."""
    newdic={}
    for key, group in itertools.groupby(dict_, lambda k: keyword \
                                        if (dict_[k]<threshold) else k):
         newdic[key] = sum([dict_[k] for k in list(group)]) 
    return newdic