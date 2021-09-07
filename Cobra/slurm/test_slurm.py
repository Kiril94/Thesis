# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 13:42:24 2021

@author: klein
"""
a = 0
for i in range(10):
    a +=i
file = open("testfile.txt","w") 
file.write(str(a))