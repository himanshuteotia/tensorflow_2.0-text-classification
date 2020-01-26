#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 22:08:04 2020

@author: himanshu
"""


import numpy as np

def NN(m1,m2,w1,w2,b):
    z = m1*m2 + w1*w2 + b
    return sigmoid(z)

def sigmoid(x):
    return 1/(1+np.exp(-x))


w1 = np.random.rand()
w2 = np.random.rand() 
b = np.random.rand()

print(NN(3,1.5,w1,w2,b))
print(NN(1,1,w1,w2,b))


#cost_function = (prediction - target)^2 (name of the function is squared error function)



