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
import matplotlib.gridspec as gridspec

#%% helper function

#optimal consumption
def integral(u, A, B, C, t):
    return np.exp(- (A*(u-t)+B*(C**u-C**t)/np.log(C)) )    

def add_new_people(id_num, A, B, C, column_name):
    x = 0
    while x < 50:
        x = np.random.normal(50,15,1)
        age0 = int(x)
        age_now = int(x)
    
    t = age_now
    
    x = random.random()
    if x < 0.5:
        sex = 0
    else:
        sex = 1
        
    mortality_rate = np.round(A[sex] + B[sex]*(C[sex]**(t)), 5)
    
    #bequest_motive
    x = np.random.randint(7)
    bequest_motive = x
    
    gamma = 0
    
    money_total = 10000
    
    r = 0
    rho = 0
    b = bequest_motive
    t = age_now
    beta = r + (rho - r) / (1 - gamma)
    lambda_t = A[sex]*t + B[sex]*C[sex]**t/np.log(C[sex])
    U = quad(integral, t, 130, args=(A[sex],B[sex],C[sex], t))
    #the later one which does not have integeraL
    later = -b**(1/(1-gamma)) * np.exp(- (A[sex]*(130-t)+B[sex]*(C[sex]**130-C[sex]**t)/np.log(C[sex]))) + \
    b**(1/(1-gamma))
    #m(b,gamma,t)
    m = U[0] + later
    ct = 1 / m
    at = 1 - min(1/m * (b**(1/(1-gamma))), 1)
    
    #optimal consumption
    optimal_consumption = ct

    #tortine proportion
    tontine_proportion = at
    
    #money_tontine
    money_tontine = money_total * tontine_proportion
    
    #money bequest
    money_bequest = money_total * (1 - tontine_proportion)
    
    return pd.DataFrame([[id_num, age0, age_now, sex, tontine_proportion, mortality_rate, gamma, \
                          money_total, money_tontine, money_bequest, bequest_motive, optimal_consumption]],
    columns = column_name)
    
#%% initialization
def wealth_monte_carlo():
    seed = 1234
    np.random.seed(1234)
    random.seed(1234)
    num_of_people = 5
    num_of_variables = 12
    
    data = np.ones([num_of_people,num_of_variables])
    
    df = pd.DataFrame(data)
    column_name = ['id', 'age0', 'age_now', 'sex', 'tontine proportion', 'mortality rate', 'gamma', 'money_total',
                     'money_tontine', 'money_bequest', 'bequest_motive', 'optimal_consumption']
    df.columns = column_name
    
    
    
    #set id
    id = list(range(1, num_of_people+1))
    df.iloc[:,0] = id
            
    for i in range(num_of_people):
        
        #initialze age
        x = 0
        while x < 50:
            x = np.random.normal(50,15,1)
            df.iloc[i,1] = int(x)
            df.iloc[i,2] = int(x)
            
    for i in range(num_of_people):
        # sex
        x = random.random()
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
    x = np.random.randint(7, size=num_of_people)
    df.iloc[:, 10] = x
    
    #gamma
    x = np.ones([num_of_people,1]) * 0
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
    wealth_track = [[10000] for i in range(8)]
    #people id 从20开始，每次死亡一人，新来的人的id加1
    people_id = num_of_people
    checked_id = list(range(1,9))
    checked_age = []
    #储存所有人进来的年龄
    for i in range(1, num_of_people+1):
        checked_age.append(df.loc[df['id']==i, 'age0'].values[0]) 
        
    initial_df = df.copy()  
    #创建一个整体dataframe的开头，然后将后续的dataframe全部整合到一起
    total_df = df.copy()
    total_df = total_df.append(pd.Series(name='year1', dtype='float64'))
    for year in range(15):
        #每年进来的时候，新来人员的名单都会清空
        people_id_group = []

        #mortality rate
        for i in range(num_of_people):
            #age now = age now + 1
            df.iloc[i,2] += 1
        
        # minus consumption money
        for i in range(num_of_people):
            df.iloc[i,7]=df.iloc[i,7]*(1-df.iloc[i,11])
            #money_tontine
            df.iloc[:,8] = df.iloc[:,7] * df.iloc[:,4]
            #money bequest
            df.iloc[:,9] = df.iloc[:,7] * (1 - df.iloc[:,4])
            
        # if death happens
        death_people = []
        for i in range(num_of_people):
            death_rate = random.random()    
            if df.iloc[i,5]>death_rate:
                donation=df.iloc[i,8]
                tontine_pool=sum(df.iloc[:,8]*df.iloc[:,5])
                for j in range(num_of_people):
                    #死亡的人将会捐出他所有的钱然后再获得分到的钱
                    if j==i:
                        df.iloc[j,7]=df.iloc[j,8]*df.iloc[j,5]*donation/tontine_pool+df.iloc[j,9]
                    else:
                        df.iloc[j,7]=df.iloc[j,8]*df.iloc[j,5]*donation/tontine_pool+df.iloc[j,7]
                    
                #每次死一个人,people id 加1
                people_id += 1
                people_id_group.append(people_id)
                death_people.append(i)
                death_id.append(df.iloc[i, 0])
                
        for i in range(1, 9):
            if df.loc[df['id']==i,'money_total'].empty==False:
                
                wealth_track[i-1].append(df.loc[df['id']==i,'money_total'].values[0])        
        ## drop the dead person
        df.drop(index = death_people, axis = 0, inplace=True)
        
        #有多少人死亡就安排多少人进来
        for i in range(len(death_people)):
            df = df.append(add_new_people(people_id_group[i], A, B, C, column_name))
            checked_age.append(df.iloc[-1,1])
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
        
            #tortine proportion
            df.iloc[i,4] = at
            
        #money_tontine
        df.iloc[:,8] = df.iloc[:,7] * df.iloc[:,4]
        
        #money bequest
        df.iloc[:,9] = df.iloc[:,7] * (1 - df.iloc[:,4])
        
        total_df = total_df.append(df)
        total_df = total_df.append(pd.Series(name='year'+str(year+2), dtype='float64'))
        total_df.to_csv('C:\\Users\\Emma\\Desktop\\REPORT\\minicase\\total_df.csv')
#        df.to_csv('C:\\Users\\Emma\\Desktop\\REPORT\\minicase\\year'+str(year+1)+'.csv')

    return wealth_track, checked_age, initial_df, df

for run in range(1):
    num_of_people = 5
    wealth, age, initial_df, df = wealth_monte_carlo()
    
    fig = plt.figure(tight_layout=True, figsize = [8,12])
    gs = gridspec.GridSpec(4, 2)
    for i in range(1):
        x = int(i / 2)
        y = int(i % 2)
        ax = fig.add_subplot(gs[x, y])
        ax.grid()
        ax.set_xlabel("Age")
        ax.set_ylabel("Total money")
        ax.set_title('The life long wealth for member '+str(i+1))
        plt.plot(list(range(int(age[i]), int(age[i])+len(wealth[i]))), wealth[i])
    
    plt.savefig('minicase.png')    
    plt.show()
    initial_df.to_csv('minicase.csv')

