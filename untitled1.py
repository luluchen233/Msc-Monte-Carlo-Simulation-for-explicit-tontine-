# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 00:46:32 2020

@author: Emma
"""


import matplotlib.pyplot as plt
plt.plot([1,2], [2,4])
plt.text(1.6, 3.5, "spam", size=25,
         ha="right", va="top",
         bbox=dict(boxstyle="square",
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),
                   )
         )
plt.show()