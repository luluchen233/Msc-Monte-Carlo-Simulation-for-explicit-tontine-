# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 19:20:43 2020

@author: Emma
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
def function(x):
    return 

def ct(x):
    a = 2.41440641e-01-7.82716468e-03*x+7.29135531e-05*x**2
    return a

def xt(x):
    return 2.96972830e+04-5.11432352e+02*x+ 2.35612061*x**2


x=np.arange(51, 100, 0.1)

plt.plot(x, ct(x))

plt.figure()
plt.plot(x, xt(x))

plt.figure()
plt.plot(x, ct(x) * xt(x))

scipy.stats.norm(3, 14).pdf(0.1)
