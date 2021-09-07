# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 17:21:34 2021

@author: klein
"""
import os
import json
import time


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

def get_running_time(start):
    m, s = divmod(time.time()-start , 60)
    h, m = divmod(m, 60)
    return f'[{h:2.0f}h{m:2.0f}m{s:2.0f}s]' 