# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 01:05:45 2020

@author: Emma
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 01:08:53 2020

@author: Emma
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 02:36:11 2020

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


#%% initialization
def wealth_monte_carlo(num_of_people, death_rate_total, bequest):
    num_of_variables = 12
    np.random.seed(30)
    
    data = np.ones([num_of_people,num_of_variables])
    
    df = pd.DataFrame(data)
    df.columns = ['id', 'age0', 'age_now', 'sex', 'tontine proportion', 'mortality rate', 'gamma', 'money_total',
                     'money_tontine', 'money_bequest', 'bequest_motive', 'optimal_consumption']
    
    
    
    #set id
    id = list(range(1, num_of_people+1))
    df.iloc[:,0] = id
            
    for i in range(num_of_people):
        
        #initialze age
        if i != 6:
            x = 0
            while x < 50:
                x = np.random.normal(50,15,1)
                df.iloc[i,1] = int(x)
                df.iloc[i,2] = int(x)
        else:
            df.iloc[i,1] = 50
            df.iloc[i,2] = 50
                
    for i in range(num_of_people):
        # sex
        x = np.random.random()
        if x < 0.5:
            df.iloc[i,3] = 0
        else:
            df.iloc[i,3] = 1
            
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
    x = np.random.randint(7, size=num_of_people-1)
    x = np.insert(x, 6, bequest)
    df.iloc[:, 10] = x
    
    #gamma
#    x = np.random.random(num_of_people) * -4 + 1
    x = np.zeros(num_of_people)
    df.iloc[:,6] = x
    
    #money total
    x = np.ones([num_of_people,1]) * 10000
    df.iloc[:,7] = x
    
    #optimal consumption
    def integral(u, A, B, C, t):
        return np.exp(- (A*(u-t)+B*(C**u-C**t)/np.log(C)) )  
    
    r = 0
    rho = 0

    for i in range(num_of_people):
        b = df.iloc[i, 10]
        t = df.iloc[i, 2]
        gamma = df.iloc[i, 6]
        beta = r + (rho - r) / (1 - df.iloc[i,6])
        lambda_t = A[df.iloc[i,3]]*t + B[df.iloc[i,3]]*C[df.iloc[i,3]]**t/np.log(C[df.iloc[i,3]])
        U = quad(integral, t, 110, args=(A[df.iloc[i,3]],B[df.iloc[i,3]],C[df.iloc[i,3]], t))
        #the later one which does not have integeraL
        later = -b**(1/(1-gamma)) * np.exp(- (A[df.iloc[i,3]]*(130-t)+B[df.iloc[i,3]]*(C[df.iloc[i,3]]**130-C[df.iloc[i,3]]**t)/np.log(C[df.iloc[i,3]]))) + \
            b**(1/(1-gamma))
        #m(b,gamma,t)
        m = U[0] + later
        ct = 1 / m
        at = 1 - min(1/m * (b**(1/(1-gamma))), 1)
        
        #optimal consumption
        df.iloc[i,-1] = ct
    
        #tortine proportion
        df.iloc[i,4] = at
        
    #money_tontine
    df.iloc[:,8] = df.iloc[:,7] * df.iloc[:,4]
    
    #money bequest
    df.iloc[:,9] = df.iloc[:,7] * (1 - df.iloc[:,4])
    
    #print(df)
    #%% loop
    death_id = []
    wealth_track = []
    round_year = -1
    while not df.loc[df['id']==7,'money_total'].empty:
        round_year += 1
#        if df.loc[df['id']==7,'money_total'].empty==False:
#            
#            wealth_track.append(df.loc[df['id']==7,'money_total'].values[0])
#        else:
#            break
        wealth_track.append(df.loc[df['id']==7,'money_total'].values[0])
        #mortality rate
        for i in range(num_of_people):
            #age now = age now + 1
            df.iloc[i,2] += 1
        
        # minus consumption money
        for i in range(num_of_people):
            df.iloc[i,7]=df.iloc[i,7]*(1-df.iloc[i,11])
        # if death happens
        death_people = []
        for i in range(num_of_people):
            death_rate = death_rate_total[round_year*num_of_people+i]   
            if df.iloc[i,5]>death_rate:
                donation=df.iloc[i,8]
                tontine_pool=sum(df.iloc[:,8]*df.iloc[:,5])
                for j in range(num_of_people):
                    if j==i:
                        df.iloc[j,7]=df.iloc[j,8]*df.iloc[j,5]*donation/tontine_pool+df.iloc[j,9]
                    else:
                        df.iloc[j,7]=df.iloc[j,8]*df.iloc[j,5]*donation/tontine_pool+df.iloc[j,7]
                    
                num_of_people-=1
                death_people.append(i)
                death_id.append(df.iloc[i, 0])
                
        ## drop the dead person
        df.drop(index = death_people, axis = 0, inplace=True)
        df.reset_index(drop=True, inplace= True)    
                
        for i in range(num_of_people):
            t = df.iloc[i,2]
            df.iloc[i,5] = np.round(A[df.iloc[i,3]] + B[df.iloc[i,3]]*(C[df.iloc[i,3]]**(t)), 5)
            
        for i in range(num_of_people):
            b = df.iloc[i, 10]
            t = df.iloc[i, 2]
            gamma = df.iloc[i, 6]
            beta = r + (rho - r) / (1 - df.iloc[i,6])
            lambda_t = A[df.iloc[i,3]]*t + B[df.iloc[i,3]]*C[df.iloc[i,3]]**t/np.log(C[df.iloc[i,3]])
            U = quad(integral, t, 110, args=(A[df.iloc[i,3]],B[df.iloc[i,3]],C[df.iloc[i,3]],t))
            #the later one which does not have integeraL
            later = -b**(1/(1-gamma)) * np.exp(- (A[df.iloc[i,3]]*(130-t)+B[df.iloc[i,3]]*(C[df.iloc[i,3]]**130-C[df.iloc[i,3]]**t)/np.log(C[df.iloc[i,3]]))) + \
            b**(1/(1-gamma))
            #m(b,gamma,t)
            m = U[0] + later
            ct = 1 / m
            at = 1 - min(1/m * (b**(1/(1-gamma))), 1)
            
            #optimal consumption
            df.iloc[i,-1] = ct
        
            #tortine proportion
            df.iloc[i,4] = at
            
        #money_tontine
        df.iloc[:,8] = df.iloc[:,7] * df.iloc[:,4]
        
        #money bequest
        df.iloc[:,9] = df.iloc[:,7] * (1 - df.iloc[:,4])
    
    return wealth_track

runtime = 1
bequest_range = [0,0.5, 1, 3,5, 7]
np.random.seed(30)
#假设进来的人最多只能活200年
num_of_people = 20
death_rate = np.random.random([num_of_people*200, runtime])
wealth_total = []
fig, ax = plt.subplots(1,1, figsize=[8,4])
for i in range(len(bequest_range)):
    b = bequest_range[i]
    for run in range(runtime):
        wealth = wealth_monte_carlo(num_of_people, death_rate[:,run], b)
        wealth_total.append(wealth)
        ax.plot(list(range(50, 50+len(wealth))), wealth, label='b='+str(b))
        print('This is ', run, ' round')
    
plt.xlabel("t=Age")
plt.grid()
plt.ylabel("Wealth ($X_t$)")  
plt.title('The life long wealth for closed tontine (b)')
plt.legend(loc='lower left')

plt.tight_layout()

SMALL_SIZE = 12
MEDIUM_SIZE = 12
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=10)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
 # 设置数据图的标题 plt.title('C语言中文网的历年销量')
plt.show()
#plt.savefig('closedsize20.eps', format='eps')
