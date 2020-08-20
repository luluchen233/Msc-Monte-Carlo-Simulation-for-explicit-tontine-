# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 20:02:37 2020

@author: Emma
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.integrate as integrate
import scipy.special as special
from scipy.integrate import quad
import matplotlib.ticker as mtick
np.random.seed(30)
num_of_people = 55
num_of_variables = 13

data = np.ones([num_of_people,num_of_variables])

df = pd.DataFrame(data)
column_name = ['id', 'age0', 'age_now', 'sex', 'tontine proportion', 'mortality rate', 'gamma', 'money_total',
                 'money_tontine', 'money_bequest', 'bequest_motive', 'optimal_consumption0','optimal_consumption7']
df.columns = column_name

def integral(u, A, B, C, t):
    return np.exp(- (A*(u-t)+B*(C**u-C**t)/np.log(C)) )  

#set id
id = list(range(1, 56))
df.iloc[:,0] = id

#initialize age
x = list(range(51, 106))
df.iloc[:,1] = x
df.iloc[:,2] = x
##initialize sex 
x = np.ones([55,1]) 
df.iloc[:, 3] = x
        
#convert float to int data type
df.iloc[:,3] = df.iloc[:,3].astype(int)
#mortality rate
A=[-0.005554283659336777, -0.004935300738226511]

B=[4.880588470507731e-05, 2.125995952047618e-05]
C=[1.09447455646221, 1.1023275011739901]



for i in range(num_of_people):
    
    t = df.iloc[i,2]
    df.iloc[i,5] = np.round(A[df.iloc[i,3]] + B[df.iloc[i,3]]*(C[df.iloc[i,3]]**(t)), 5)
    
#bequest_motive
x = np.zeros([55,1]) 
df.iloc[:, 10] = x

#gamma
x = np.ones([55,1]) * 0
df.iloc[:,6] = x

#money total
x = np.ones([55,1]) * 10000
df.iloc[:,7] = x

r = 0
rho = 0

for i in range(num_of_people):
    b = df.iloc[i, 10]
    t = df.iloc[i, 2]
    gamma = df.iloc[i, 6]
    beta = r + (rho - r) / (1 - df.iloc[i,6])
    lambda_t = A[df.iloc[i,3]]*t + B[df.iloc[i,3]]*C[df.iloc[i,3]]**t/np.log(C[df.iloc[i,3]])
    U = quad(integral, t, 130, args=(A[df.iloc[i,3]],B[df.iloc[i,3]],C[df.iloc[i,3]], t))
    #the later one which does not have integeraL
    later = -b**(1/(1-gamma)) * np.exp(- (A[df.iloc[i,3]]*(130-t)+B[df.iloc[i,3]]*(C[df.iloc[i,3]]**130-C[df.iloc[i,3]]**t)/np.log(C[df.iloc[i,3]]))) + \
        b**(1/(1-gamma))
    #m(b,gamma,t)
    m = U[0] + later
    ct = 1 / m
    at = 1 - min(1/m * (b**(1/(1-gamma))), 1)
    
    #optimal consumption
    df.iloc[i,-2] = ct

    #tortine proportion
    df.iloc[i,4] = at

ct_track = []
for i in range(num_of_people):
    b = 3
    t = df.iloc[i, 2]
#    gamma = df.iloc[i, 6]
    gamma = 0
    beta = r + (rho - r) / (1 - df.iloc[i,6])
    lambda_t = A[df.iloc[i,3]]*t + B[df.iloc[i,3]]*C[df.iloc[i,3]]**t/np.log(C[df.iloc[i,3]])
    U = quad(integral, t, 130, args=(A[df.iloc[i,3]],B[df.iloc[i,3]],C[df.iloc[i,3]], t))
    #the later one which does not have integeraL
    later = -b**(1/(1-gamma)) * np.exp(- (A[df.iloc[i,3]]*(130-t)+B[df.iloc[i,3]]*(C[df.iloc[i,3]]**130-C[df.iloc[i,3]]**t)/np.log(C[df.iloc[i,3]]))) + \
        b**(1/(1-gamma))
    #m(b,gamma,t)
    m = U[0] + later
    ct = 1 / m
    at = 1 - min(1/m * (b**(1/(1-gamma))), 1)
    
    #optimal consumption
    df.iloc[i,-1] = ct
    
    ct_track.append(ct)
    

plt.plot(ct_track)

#b_range = [0,1,3,5,7]
#gamma_range = [0.99,0.5,0,-1,-2,-3]
#fig, ax = plt.subplots(3,2,figsize=[12,60])
#for i in range(6):
#    gamma = gamma_range[i]
#    x = int(i / 2)
#    y = int(i % 2)
#    for b in b_range:
#        ct_track = []
#        for i in range(num_of_people):
#            t = df.iloc[i, 2]
#            U = quad(integral, t, 130, args=(A[df.iloc[i,3]],B[df.iloc[i,3]],C[df.iloc[i,3]], t))
#            #the later one which does not have integeraL
#            later = -b**(1/(1-gamma)) * np.exp(- (A[df.iloc[i,3]]*(130-t)+B[df.iloc[i,3]]*(C[df.iloc[i,3]]**130-C[df.iloc[i,3]]**t)/np.log(C[df.iloc[i,3]]))) + \
#                b**(1/(1-gamma))
#            #m(b,gamma,t)
#            m = U[0] + later
#            ct = 1 / m
#            ct_track.append(ct)
#        ax[x,y].plot(list(range(51, 106)), np.array(ct_track), label='b='+str(b))
#    ax[x,y].legend()
#    ax[x,y].set_xlabel('t=age')
#    ax[x,y].set_ylabel('$c_t^{*}$    ',rotation=0)
#    ax[x,y].grid()
#    if x==0 and y==0:
#        ax[x,y].set_title('Optimal consumption rate, UK male, $\gamma$='+str(1))
#    else:
#        ax[x,y].set_title('Optimal consumption rate, UK male, $\gamma$='+str(gamma))

plt.tight_layout()
plt.show()
SMALL_SIZE = 13
MEDIUM_SIZE = 13
BIGGER_SIZE = 13

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
 # 设置数据图的标题 plt.title('C语言中文网的历年销量')
plt.show()

#%% comparison of ct and at
def plot_figure():
    #m(0,gamma,t)
    ex = pd.read_csv("qx.csv", usecols=[3])
    b = 3
    ct = (ex+b)**-1
    alpha = 1 - b*ct
    plt.plot(range(50,101), alpha.iloc[50:101,0])

from scipy import optimize

def helper():
    xdata =  np.arange(51,106)
    ydata = np.array(ct_track)
    params, params_covariance = optimize.curve_fit(func, xdata, ydata)
    
    plt.plot(xdata, func(xdata, params[0], params[1], params[2]), label='Fit')
    plt.plot(xdata, ydata, label='train')
    plt.legend()
    plt.title('The parameter fitting for wealth (b=3, $\gamma$=0)')
    plt.xlabel('t=Age')
    plt.ylabel('Wealth ($X_t$)')
    plt.grid()
    
    print(params)
    
def func(x,a,b,c):
    return a + b*x + c*x*x
#    return a + b*np.exp(x)