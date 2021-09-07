# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 17:21:34 2021

@author: klein
"""
import os
import json


def count_subdirectories(dir_):
    """Counts level 1 subdirectories"""
    return sum(os.path.isdir(os.path.join(dir_,x)) for x \
               in os.listdir(dir_))

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
def list_subdir(dir_):
    return [os.path.join(dir_, x) for x in os.listdir(dir_)]

def get_json(path):
    """Returns data, contained in a json file under path."""
    with open(path, 'r') as f:
    	data = json.load(f)
    return data

def create_dictionary(keys, values):
    result = {} # empty dictionary
    for key, value in zip(keys, values):
        result[key] = value
    return result