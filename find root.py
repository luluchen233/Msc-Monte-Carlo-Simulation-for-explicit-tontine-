# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 00:37:03 2020

@author: Emma
"""

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
def func(x):
    return (2.06839853e-01 + 6.86472771e-03*x + 6.75373135e-05*x*x) * (2.96972830e+04 - 5.11432352e+02*x + 2.35612061*x*x)
    
x = np.array([-100,100])
print(optimize.fsolve(func,x))

plt.plot(np.arange(0,100,0.1), func(np.arange(0,100,0.1)))