# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 20:18:06 2019

@author: ckc
"""

def f(output_size=None ,ksize=None,stride=None):
    perceptive_filed = (output_size - 1) * stride + ksize
    return perceptive_filed

conv_2 = f(f(f(f(f(f(f(f(f(f(f(f(1,3,1),5,1),3,1),3,1),3,1),3,1),3,1),3,1),3,1),3,1),5,1),3,1)
print(conv_2)

simulate_signal = [2*i for i in [2,3]]

