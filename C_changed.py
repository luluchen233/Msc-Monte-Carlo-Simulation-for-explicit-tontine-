# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 14:16:39 2020

@author: Emma
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 11:10:02 2020

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
import matplotlib.gridspec as gridspec
import timeit

#%% helper function

#optimal consumption
def integral(u, A, B, C, t):
    return np.exp(- (A*(u-t)+B*(C**u-C**t)/np.log(C)) )    

def add_new_people(id_num, A, B, C, column_name):
#    x = 0
#    while x < 50:
#        x = np.random.normal(50,15,1)
    age0 = 50
    age_now = age0
    
    t = age_now
    
#    x = random.random()
#    if x < 0.5:
#        sex = 0
#    else:
#        sex = 1
    sex=0
        
    mortality_rate = np.round(A[sex] + B[sex]*(C[sex]**(t)), 5)
    
    #bequest_motive
#    x = np.random.randint(7)
    bequest_motive = 3
    
    gamma = 0
    
    money_total = 10000
    
    r = 0
    rho = 0
    b = bequest_motive
    t = age_now
#    beta = r + (rho - r) / (1 - gamma)
#    lambda_t = A[sex]*t + B[sex]*C[sex]**t/np.log(C[sex])
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
def wealth_monte_carlo(num_of_people, term_C):
    start = timeit.default_timer() 
#    np.random.seed(1234)
#    random.seed(100)
    num_of_variables = 12
    
    data = np.ones([num_of_people,num_of_variables])
    
    df = pd.DataFrame(data)
    column_name = ['id', 'age0', 'age_now', 'sex', 'tontine proportion', 'mortality rate', 'gamma', 'money_total',
                     'money_tontine', 'money_bequest', 'bequest_motive', 'optimal_consumption']
    df.columns = column_name
    
    
    
    #set id
    id = list(range(1, num_of_people+1))
    df.iloc[:,0] = id  
#    for i in range(num_of_people):
#        
#        #initialze age
#        x = 0
#        while x < 50:
#            x = np.random.normal(50,15,1)
#            df.iloc[i,1] = int(x)
#            df.iloc[i,2] = int(x)
    
    df.iloc[:,1] = np.ones([num_of_people]) * 50
    df.iloc[:,2] = np.ones([num_of_people]) * 50
#    age_list = np.random.randint(50,80,size=num_of_people-1)
#    age_list = np.insert(age_list,0,50)
#    df.iloc[:,1] = age_list
#    df.iloc[:,2] = age_list
     
#    for i in range(num_of_people):
        # sex
#        x = random.random()
#        if x < 0.5:
#            df.iloc[i,3] = 0
#        else:
#            df.iloc[i,3] = 1
    df.iloc[:,3] = np.zeros([num_of_people])
 
    #convert float to int data type
    df.iloc[:,3] = df.iloc[:,3].astype(int)
    #mortality rate
    A=np.array([[-0.005554283659336777, -0.004935300738226511]]*num_of_people)
    
    B=np.array([[4.880588470507731e-05, 2.125995952047618e-05]]*num_of_people)
    C=np.array([[term_C, 1.1023275011739901]]*num_of_people)
      
#    for i in range(num_of_people):
#        
#        t = df.iloc[i,2]
#        df.iloc[i,5] = np.round(A[df.iloc[i,3]] + B[df.iloc[i,3]]*(C[df.iloc[i,3]]**(t)), 5)
    df.iloc[:,5] = np.round(A[range(num_of_people), df.iloc[:,3]] + B[range(num_of_people), df.iloc[:,3]]*
                   (C[range(num_of_people), df.iloc[:,3]]**(df.iloc[:,2])), 5)
        
    #bequest_motive
#    x = np.random.randint(7, size=num_of_people)
    df.iloc[:, 10] = np.ones(num_of_people) * 3
    
    #gamma
    x = np.ones([num_of_people,1]) * 0
    df.iloc[:,6] = x
    
    #money total
    x = np.ones([num_of_people,1]) * 10000
    df.iloc[:,7] = x
    
    r = 0
    rho = 0

#    for i in range(num_of_people):
#        b = df.iloc[i, 10]
#        t = df.iloc[i, 2]
#        gamma = df.iloc[i, 6]
#        beta = r + (rho - r) / (1 - df.iloc[i,6])
#        lambda_t = A[df.iloc[i,3]]*t + B[df.iloc[i,3]]*C[df.iloc[i,3]]**t/np.log(C[df.iloc[i,3]])
#        U = quad(integral, t, 130, args=(A[df.iloc[i,3]],B[df.iloc[i,3]],C[df.iloc[i,3]], t))
#        #the later one which does not have integeraL
#        later = -b**(1/(1-gamma)) * np.exp(- (A[df.iloc[i,3]]*(130-t)+B[df.iloc[i,3]]*(C[df.iloc[i,3]]**130-C[df.iloc[i,3]]**t)/np.log(C[df.iloc[i,3]]))) + \
#            b**(1/(1-gamma))
#        #m(b,gamma,t)
#        m = U[0] + later
#        ct = 1 / m
#        at = 1 - min(1/m * (b**(1/(1-gamma))), 1)
#        
#        #optimal consumption
#        df.iloc[i,-1] = ct
#    
#        #tortine proportion
#        df.iloc[i,4] = at
        
    set_u = np.zeros([num_of_people])
    for i in range(num_of_people):
            U = quad(integral, df.iloc[i,2], 130, args=(A[0,df.iloc[i,3]],B[0,df.iloc[i,3]],C[0,df.iloc[i,3]], df.iloc[i,2]))
            set_u[i] = U[0]
        
    later = -df.iloc[:,10]**(1/(1-df.iloc[:,6])) * np.exp(- (A[range(num_of_people), df.iloc[:,3]]*(130-df.iloc[:,2])+\
                        B[range(num_of_people),df.iloc[:,3]]*(C[range(num_of_people),df.iloc[:,3]]**130-C[range(num_of_people),df.iloc[:,3]]**
                         df.iloc[:,2])/np.log(C[range(num_of_people),df.iloc[:,3]]))) + df.iloc[:,10]**(1/(1-df.iloc[:,6]))
    #optimal consumption
    df.iloc[:,-1] = 1 / (set_u + later)
    #tortine proportion
    df.iloc[:,4] = 1 - np.minimum(1/(set_u+later) * (df.iloc[:,10]**(1/(1-df.iloc[:,6]))), np.ones([num_of_people]))
        
    #money_tontine
    df.iloc[:,8] = df.iloc[:,7] * df.iloc[:,4]
    
    #money bequest
    df.iloc[:,9] = df.iloc[:,7] * (1 - df.iloc[:,4])
    
    #print(df)
    #%% loop
    death_id = []
    wealth_track = [10000]
    ct_track = [df.loc[df['id']==1, 'optimal_consumption'].values[0]]
    utontine_track = [np.log(df.loc[df['id']==1, 'optimal_consumption'].values[0] * df.loc[df['id']==1, 'money_total'].values[0])]
    ubequest_track = [df.loc[df['id']==1, 'bequest_motive'].values[0] * df.loc[df['id']==1, 'mortality rate'].values[0] * 
                np.log((1-df.loc[df['id']==1, 'tontine proportion'].values[0])*df.loc[df['id']==1, 'money_total'].values[0])]
    #people id 从20开始，每次死亡一人，新来的人的id加1
    people_id = num_of_people
    checked_id = list(range(1,9))
    checked_age = []
    xt_mul_ct = [df.iloc[0,-1]*df.iloc[0,7]]
    #储存所有人进来的年龄
    for i in range(1, 9):
        checked_age.append(df.loc[df['id']==i, 'age0'].values[0]) 
        
    initial_df = df.copy()  
    #创建一个整体dataframe的开头，然后将后续的dataframe全部整合到一起
    total_df = df.copy()
    total_df = total_df.append(pd.Series(name='year1', dtype='float64'))
    
    stop = timeit.default_timer()

    for year in range(50):
        start1 = timeit.default_timer()
        #每年进来的时候，新来人员的名单都会清空
        people_id_group = []

        #age now = age now + 1
        df.iloc[:,2] += 1
        
        # minus consumption money
        df.iloc[:,7]=df.iloc[:,7]*(1-df.iloc[:,11])
        #money_tontine
        df.iloc[:,8] = df.iloc[:,7] * df.iloc[:,4]
        #money bequest
        df.iloc[:,9] = df.iloc[:,7] * (1 - df.iloc[:,4])
            
        # if death happens
        death_people = []
        start3 = timeit.default_timer()
        for i in range(num_of_people):
            if i != 0:
                death_rate = random.random()    
            else:
                death_rate = 1
            if df.iloc[i,5]>death_rate:
                donation=df.iloc[i,8]
                tontine_pool=sum(df.iloc[:,8]*df.iloc[:,5])
                df.iloc[:,7]=df.iloc[:,8]*df.iloc[:,5]*donation/tontine_pool+df.iloc[:,7]
                #减去不该加的加上该加的
                df.iloc[i,7] = df.iloc[i,8]*df.iloc[i,5]*donation/tontine_pool+df.iloc[i,9]
#                for j in range(num_of_people):
#                    #死亡的人将会捐出他所有的钱然后再获得分到的钱
#                    if j==i:
#                        df.iloc[j,7]=df.iloc[j,8]*df.iloc[j,5]*donation/tontine_pool+df.iloc[j,9]
#                    else:
#                        df.iloc[j,7]=df.iloc[j,8]*df.iloc[j,5]*donation/tontine_pool+df.iloc[j,7]
                    
                #每次死一个人,people id 加1
                people_id += 1
                people_id_group.append(people_id)
                death_people.append(i)
                death_id.append(df.iloc[i, 0])
                
        stop3 = timeit.default_timer()
        
#        for i in range(1,2):
#            if df.loc[df['id']==i,'money_total'].empty==False:
#                
#                wealth_track[i-1].append(df.loc[df['id']==i,'money_total'].values[0])
                
        ## drop the dead person
        df.drop(index = death_people, axis = 0, inplace=True)
        
        #有多少人死亡就安排多少人进来
        for i in range(len(death_people)):
            df = df.append(add_new_people(people_id_group[i], A[0], B[0], C[0], column_name))
            checked_age.append(df.iloc[-1,1])
                
        df.reset_index(drop=True, inplace= True)    
        
        start4 = timeit.default_timer()
        df.iloc[:,5] = np.round(A[range(num_of_people), df.iloc[:,3]] + B[range(num_of_people), df.iloc[:,3]]*
                   (C[range(num_of_people), df.iloc[:,3]]**(df.iloc[:,2])), 5)
            
        stop4 = timeit.default_timer()
        
            
        start5 = timeit.default_timer()
        set_u = np.zeros([num_of_people])
#        for i in range(num_of_people):
#            b = df.iloc[i, 10]
#            t = df.iloc[i, 2]
#            gamma = df.iloc[i, 6]
##            beta = r + (rho - r) / (1 - df.iloc[i,6])
##            lambda_t = A[df.iloc[i,3]]*t + B[df.iloc[i,3]]*C[df.iloc[i,3]]**t/np.log(C[df.iloc[i,3]])
#            U = quad(integral, t, 130, args=(A[0,df.iloc[i,3]],B[0,df.iloc[i,3]],C[0,df.iloc[i,3]], t))
#            #the later one which does not have integeraL
#            later = -b**(1/(1-gamma)) * np.exp(- (A[0,df.iloc[i,3]]*(130-t)+B[0,df.iloc[i,3]]*(C[0,df.iloc[i,3]]**130-C[0,df.iloc[i,3]]**t)/np.log(C[0,df.iloc[i,3]]))) + \
#                b**(1/(1-gamma))
#            #m(b,gamma,t)
#            m = U[0] + later
#            ct = 1 / m
#            at = 1 - min(1/m * (b**(1/(1-gamma))), 1)
#            
#            #optimal consumption
#            df.iloc[i,-1] = ct
#        
#            #tortine proportion
#            df.iloc[i,4] = at
        for i in range(num_of_people):
            U = quad(integral, df.iloc[i,2], 130, args=(A[0,df.iloc[i,3]],B[0,df.iloc[i,3]],C[0,df.iloc[i,3]], df.iloc[i,2]))
            set_u[i] = U[0]
            
        later = -df.iloc[:,10]**(1/(1-df.iloc[:,6])) * np.exp(- (A[range(num_of_people), df.iloc[:,3]]*(130-df.iloc[:,2])+\
                        B[range(num_of_people),df.iloc[:,3]]*(C[range(num_of_people),df.iloc[:,3]]**130-C[range(num_of_people),df.iloc[:,3]]**
                         df.iloc[:,2])/np.log(C[range(num_of_people),df.iloc[:,3]]))) + df.iloc[:,10]**(1/(1-df.iloc[:,6]))
        #optimal consumption
        df.iloc[:,-1] = 1 / (set_u + later)
        #tortine proportion
        df.iloc[:,4] = 1 - np.minimum(1/(set_u+later) * (df.iloc[:,10]**(1/(1-df.iloc[:,6]))), np.ones([num_of_people]))
        stop5 = timeit.default_timer()

        #money_tontine
        df.iloc[:,8] = df.iloc[:,7] * df.iloc[:,4]
        
        #money bequest
        df.iloc[:,9] = df.iloc[:,7] * (1 - df.iloc[:,4])
        
#        for i in range(1,2):
#            if df.loc[df['id']==i,'money_total'].empty==False:
#                
#                wealth_track[i-1].append(df.loc[df['id']==i,'money_total'].values[0])
#                ct_track.append(df.loc[df['id']==i,'optimal_consumption'].values[0])
        for i in range(1,2):        
            if df.loc[df['id']==1,'money_total'].empty==False:
                xt_mul_ct.append(df.loc[df['id']==1,'money_total']*df.loc[df['id']==1, 'optimal_consumption'])
                ct_track.append(df.loc[df['id']==i,'optimal_consumption'].values[0])
                wealth_track.append(df.loc[df['id']==i,'money_total'].values[0])
                utontine_track.append(np.log(df.loc[df['id']==1, 'optimal_consumption'].values[0] * df.loc[df['id']==1, 'money_total'].values[0]))
                ubequest_track.append(df.loc[df['id']==1, 'bequest_motive'].values[0] * df.loc[df['id']==1, 'mortality rate'].values[0] * 
                            np.log((1-df.loc[df['id']==1, 'tontine proportion'].values[0])*df.loc[df['id']==1, 'money_total'].values[0]))
#                ct_track.append(df.loc[df['id']==i,'optimal_consumption'].values[0])
#        total_df = total_df.append(df)
#        total_df = total_df.append(pd.Series(name='year'+str(year+2), dtype='float64'))
#        total_df.to_csv('C:\\Users\\Emma\\Desktop\\REPORT\\minicase\\total_df.csv')
#        df.to_csv('C:\\Users\\Emma\\Desktop\\REPORT\\minicase\\year'+str(year+1)+'.csv')
        stop1 = timeit.default_timer()
        
        
#        print('first loop', stop3 - start3)
#        print('second loop', stop4 - start4)
#        print('third loop', stop5-start5)
#        print("total:", stop1-start1)
#        print('This is ',year, ' year')
#    print('initial part ', stop - start) 
#    plt.plot(xt_mul_ct)
#    return wealth_track, checked_age, initial_df, df
    return wealth_track, ct_track, utontine_track, ubequest_track

wealth_total = []
ct_total = []
utontine_total = []
ubequest_total = []

wealth_total1 = []
ct_total1 = []
utontine_total1 = []
ubequest_total1 = []

runtime = 20
plt.figure(figsize=[8,5])
C = [1.105, 1.09447455646221]
for i in range(2):
    for run in range(runtime):
        print('round', run)
        num_of_people = 5000
    
        wealth_track, ct_track, utontine, ubequest = wealth_monte_carlo(num_of_people, C[i])
        if i == 0:
            wealth_total.append(wealth_track)
            ct_total.append(ct_track)
            utontine_total.append(utontine)
            ubequest_total.append(ubequest)
        else:
            wealth_total1.append(wealth_track)
            ct_total1.append(ct_track)
            utontine_total1.append(utontine)
            ubequest_total1.append(ubequest)
        
from scipy import optimize

def helper():
    plt.figure(figsize=[8,5])
    data = pd.DataFrame(wealth_total)
    sum_col = data.sum()
    count = data.count()
    data_mod = sum_col / count
    
    xdata =  np.arange(50,50+len(data_mod))
    ydata = data_mod.to_numpy()
    params, params_covariance = optimize.curve_fit(func, xdata, ydata)
    
#    plt.plot(list(range(50, 50+len(wealth_track))), wealth_track[0]*ct_track[0] / ct_track, label='Theoretical')
    plt.plot(xdata, func(xdata, params[0], params[1], params[2]), label='Fit')
    plt.plot(xdata, data_mod, label = 'Cluster')
    plt.legend()
    plt.title('The theoretical and real wealth comparison (b=3, $\gamma$=0)')
    plt.xlabel('t=Age')
    plt.ylabel('Wealth ($X_t$)')
    plt.grid()
    
    plt.tight_layout()

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
    print(params)
def func(x,a,b,c):
    return a + b*x + c*x*x

def plot_wealth(x):
    fig,ax = plt.subplots(1,1,figsize=[7,5])
    for i in range(len(wealth_total)):
        ax.plot(list(range(50, 50+len(x[i]))), np.array(x[i]))
    
    plt.title('Total wealth (Increased mortality)')
    plt.xlabel('t=Age')
    plt.ylabel('Wealth ($X_t$)')
    plt.grid()
    plt.vlines(80, 0, 10000, linestyles='dashed', colors = 'indianred')
    plt.tight_layout()

    SMALL_SIZE = 17
    MEDIUM_SIZE = 17
    BIGGER_SIZE = 17
    
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    plt.show()
    
def plot_ct():
    fig,ax = plt.subplots(1,1,figsize=[8,5])
    ax.plot(list(range(50, 50+len(ct_total[0]))), np.array(ct_total[0]), label ='Increased mortality')
    ax.plot(list(range(50, 50+len(ct_total1[0]))), np.array(ct_total1[0]), label ='Normal mortality')
    
    plt.title('Optimal fractional consumption rate comparison ($\gamma=0$, b=3, Size=5,000)')
    plt.xlabel('t=Age')
    plt.ylabel('$c_t^{*}$        ',rotation=0)
    plt.grid()
    plt.legend(loc='lower right')
    plt.tight_layout()

    SMALL_SIZE = 13
    MEDIUM_SIZE = 13
    BIGGER_SIZE = 13
    
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=10)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    plt.show()

def plot_ctxt(x,y):
    fig,ax = plt.subplots(1,1,figsize=[8,5])
    for i in range(len(wealth_total)):
        ax.plot(list(range(50, 50+len(x[i]))), np.array(x[i])*np.array(y[i]))
    
    plt.title('Monetary consumption (Normal mortality, $\gamma=0$, b=3, Size=5,000)')
#    plt.title('Monetary consumption (Increased mortality, $\gamma=0$, b=3, Size=5,000)')
    plt.xlabel('t=Age')
    plt.ylabel('$c_t^{*}X_t$        ',rotation=0)
    plt.grid()
#    plt.vlines(70, 150, 375, linestyles='dashed', colors = 'indianred')
    plt.vlines(80, 175, 300, linestyles='dashed', colors = 'indianred')
    plt.tight_layout()

    SMALL_SIZE = 13
    
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title
    
    plt.show()
data = pd.DataFrame(wealth_total)
sum_col = data.sum()
count = data.count()
data_mod = sum_col / count

xdata =  np.arange(50,50+len(data_mod))
ydata = data_mod.to_numpy()
params, params_covariance = optimize.curve_fit(func, xdata, ydata)

fit_data = func(xdata, params[0], params[1], params[2])
idx = 0
for i in range(runtime):
    if len(ct_total[i]) > len(ct_total[idx]):
        idx = i
        
plt.plot(xdata, fit_data * ct_total[idx])
ct_params, ct_params_covariance = optimize.curve_fit(func, xdata, np.array(ct_total[idx]))
plt.draw()
plt.grid()
plt.xlabel('t=Age') 
plt.ylabel('$c_t^{*}X_t$       ', rotation=0)
plt.title('The monetary consumption for fitted $X_t$')

plt.tight_layout()

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