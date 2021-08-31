# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 17:21:34 2021

@author: klein
"""
import os


def count_subdirectories(dir_):
    return sum(os.path.isdir(os.path.join(dir_,x)) for x \
               in os.listdir(dir_))

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
def list_subdir(dir_):
    return [os.path.join(dir_, x) for x in os.listdir(dir_)]