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

#%% comparison of ct and at
def plot_figure():
    #m(0,gamma,t)
    ex = pd.read_csv("qx.csv", usecols=[3])
    b = 3
    ct = (ex+b)**-1
    alpha = 1 - b*ct
    plt.plot(range(50,101), ct.iloc[50:101,0], label="test")
    plt.plot(range(50,101), alpha.iloc[50:101,0], label="alpha test")
    
#%%
np.random.seed(30)
num_of_people = 50
num_of_variables = 13

data = np.ones([num_of_people,num_of_variables])

df = pd.DataFrame(data)
column_name = ['id', 'age0', 'age_now', 'sex', 'tontine proportion', 'mortality rate', 'gamma', 'money_total',
                 'money_tontine', 'money_bequest', 'bequest_motive', 'optimal_consumption0','optimal_consumption7']
df.columns = column_name


def integral(u, A, B, C, t):
    return np.exp(- (A*(u-t)+B*(C**u-C**t)/np.log(C)) )  

#set id
id = list(range(1, num_of_people+1))
df.iloc[:,0] = id

#initialize age
x = list(range(51, 51+num_of_people))
df.iloc[:,1] = x
df.iloc[:,2] = x
##initialize sex 
x = np.zeros([num_of_people,1]) 
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
x = np.ones([num_of_people,1]) *3
df.iloc[:, 10] = x

#gamma
x = np.ones([num_of_people,1]) * (-999999)
df.iloc[:,6] = x

#money total
x = np.ones([num_of_people,1]) * 10000
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

    
#money_tontine
df.iloc[:,8] = df.iloc[:,7] * df.iloc[:,4]

#money bequest
df.iloc[:,9] = df.iloc[:,7] * (1 - df.iloc[:,4])


plt.plot(list(range(50,50+num_of_people)), df['optimal_consumption0'], label='train') 
plt.plot(list(range(50,50+num_of_people)), df['tontine proportion'], label='train alpha')
plot_figure()
plt.legend()
plt.grid()
plt.xlabel('t=age') 
plt.ylabel('$c_t^x$        ',rotation=0)
plt.title('The optimal consumption rate for male')

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

