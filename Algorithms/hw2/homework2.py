# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 16:36:29 2019

@author: Egor Turukhanov
"""

import numpy as np
import matplotlib.pyplot as plt
%matplotlib qt
plt.style.use('seaborn')
import random
import time
from datetime import datetime
import pandas as pd
import timeit
import random

from math import fabs

############# Number ONE ##################3

def first(x):
    return x**3


def second(x):
    return fabs(x-0.2)

def third(x):
    return x*np.sin(1/x)

xs = np.arange(0,1.001, 0.001)

def plot_func(func, data, title):
    result =list()
    for element in data:
        result.append(func(element))
    plt.plot(data, result, 'b')
    plt.title('{}'.format(title))
    plt.show()

plot_func(third, xs[1:], 'Function is X * Sin(1/X')

# Exh Search func
def ex_search(func, xs):
    mins = list()
    iterations=len(xs)
    for x in xs:
        mins.append([x,func(x)])
    print('Minimum of f(X): ', min(mins), iterations)

ex_search(first, xs)
ex_search(second, xs)
ex_search(third, xs[1:])

# Dihotomy func
     
def dihotomy(func, xs):
    lower = xs[0]
    upper = xs[-1]
    iteration = 0 
    while fabs(lower - upper) > 0.001:
        iteration += 1
        x1 = (lower + upper - 0.0005)/2
        x2 = (lower + upper + 0.0005)/2
        #val = array[x]
        if func(x1) <= func(x2):
            lower = lower
            upper = x2
            #return x
        elif func(x1) >= func(x2):
            lower = x1
            upper = upper
        #print(lower, ' ', upper)
    return lower, upper, iteration
            
dihotomy(first, xs)
dihotomy(second, xs)
dihotomy(third, xs[1:])


# Golden ratio
def golden_section(func, xs):
    lower = xs[0]
    upper = xs[-1]
    iteration = 0 
    while fabs(lower - upper) > 0.001:
        iteration += 1
        x1 = lower + ((3-np.sqrt(5))*(upper-lower)/2)
        x2 = upper + ((np.sqrt(5)-3)*(upper-lower)/2)
        #val = array[x]
        if func(x1) <= func(x2):
            lower = lower
            upper = x2
            x2 = x1
            #return x
        elif func(x1) >= func(x2):
            lower = x1
            upper = upper
            x1=x2
        #print(lower, ' ', upper)
    return lower, upper, iteration
            
golden_section(first, xs)
golden_section(second, xs)
golden_section(third, xs[1:])
    
############# Second ###############

alpha = random.uniform(0, 1)
beta = random.uniform(0, 1)

noisy_dt= list()
def noisy(al, bt, elements):
    for i in range(0, elements,1):
        x=i/1000
        y=al*x+bt+np.random.normal(0,1)
        noisy_dt.append([x,y])
        
noisy(alpha, beta, 1000)

def F_linear(x, a, b):
    return a*x+b
    
def F_rational(x, a, b):
    return a/(1+b*x)       

def lsq_lin(x):
    d=0
    for i in range(len(noisy_dt)):
       d = d+((x[0]*noisy_dt[i][0]+x[1]) - noisy_dt[i][1])**2
    return d


#Gradient Descent
    
# Conjugate Gradient Descent
# Newton’s method
# Levenberg-Marquardt



### Nelder Mead    
from scipy import optimize
nelder_mead = optimize.minimize(lsq_lin, [0 ,1], method='Nelder-Mead')
############################# Linear ##############################
######### Exh search
a = np.arange(0.1,0.9, 0.001)
b = np.arange(0, 1, 0.001)
xs_1 = np.array([a, b])
xs_1[0][0]

a = np.arange(0.1,0.9, 0.001)
b = np.arange(0, 1, 0.001)

a_list = np.arange(0.1,0.9, 0.001)
b_list = np.arange(0, 1, 0.001)

def lsq_lin(a,b):
    d=0
    for i in range(len(noisy_dt)):
       d = d+((a*noisy_dt[i][0]+b) - noisy_dt[i][1])**2
    return d

d_list = list()
def exh_search():
    for a in a_list:
        for b in b_list:
            d_list.append([lsq_lin(a,b),a,b])
    return min(d_list)        
        
exh_sch = exh_search()        

        
####### Gauss
#a_list = np.arange(0.1,0.3, 0.001)
#b_list = np.arange(1,1.20, 0.001)

def lsq_lin(a,b):
    d=0
    for i in range(len(noisy_dt)):
       d = d+((a*noisy_dt[i][0]+b) - noisy_dt[i][1])**2
    return d

d_list = list()
list_a = list()
list_b = list()


def iteration_i():
    for a in a_list:
        list_a.append([lsq_lin(a,b_list[0]),a])
    a_opt=min(list_a)[1]
    for b in b_list:
        list_b.append([lsq_lin(a_opt,b),b])
    b_opt=min(list_b)[1]
    return a_opt, b_opt

def iteration_opt():
    for a in a_list:
        list_a.append([lsq_lin(a,b_opt),b_opt])
    a_opt=min(list_a)[1]
    for b in b_list:
        list_b.append([lsq_lin(a_opt,b),b])
    b_opt=min(list_b)[1]
    return a_opt, b_opt



def gauss():
    
    for i in range(0,100,1):
        if fabs(d_list[-1][0] - d_list[-2][0]) > 0.001:
            if i ==0:
                a_opt,b_opt = iteration_i()
            else:
                a_opt,b_opt = iteration_opt()
            
            #i += 1
            d_list.append([lsq_lin(a_opt,b_opt),a_opt,b_opt])  
        elif fabs(d_list[-1][0] - d_list[-2][0]) < 0.001:
            return (d_list[-1])
            break
       
gauss = gauss()        


d_list.append([lsq_lin(0,1),0,1])
d_list.append([lsq_lin(0,20),0,20])

##################### Plotting ################3

# Data plotting
noisy_df = pd.DataFrame(noisy_dt, columns=['x', 'y'])
plt.plot(noisy_df.x, noisy_df.y, 'ob')
plt.title('Noisy data')
plt.show()
#predicted_lin = predicted_lin.append({'y_pred' = 1})

predicted_lin = noisy_df.copy()
predicted_lin.index = predicted_lin.x
predicted_lin.index[0]
pred_mead = list()
pred_exh = list()
pred_gauss = list()



for i in range(len(noisy_dt)):
    pred_mead.append(F_linear(predicted_lin.index[i], nelder_mead['x'][0], nelder_mead['x'][1]))
for i in range(len(noisy_dt)):
    pred_exh.append(F_linear(predicted_lin.index[i], exh_sch[1], exh_sch[2]))
for i in range(len(noisy_dt)):
    pred_gauss.append(F_linear(predicted_lin.index[i], gauss[1], gauss[2]))
    

pred_mead = pd.Series(pred_mead)
pred_exh = pd.Series(pred_exh)
pred_gauss = pd.Series(pred_gauss)


predicted_lin['y_pred_mead'] = pred_mead.values
predicted_lin['y_pred_exh'] = pred_exh.values
predicted_lin['y_pred_gauss'] = pred_gauss.values


predicted_lin.drop('x', axis=1, inplace=True)

plt.scatter(noisy_df.x,noisy_df.y)
plt.plot([min(predicted_lin.index),max(predicted_lin.index)], 
          [min(predicted_lin.y_pred_gauss),max(predicted_lin.y_pred_gauss)], 'b', label='Gauss Regression line')
plt.plot([min(predicted_lin.index),max(predicted_lin.index)], 
          [min(predicted_lin.y_pred_mead),max(predicted_lin.y_pred_mead)], 'r', label='Nelder-Mead Regression line')
plt.plot([min(predicted_lin.index),max(predicted_lin.index)], 
          [min(predicted_lin.y_pred_exh),max(predicted_lin.y_pred_exh)], 'y', label='Exh Search Regression line')

plt.title('Regression line comparasion for linear function')
plt.legend()
plt.show()




############################# Rational ##############################
def F_rational(x, a, b):
    return a/(1+b*x)       

def rational(x):
    d=0
    for i in range(len(noisy_dt)):
       d = d+((x[0]/(1+x[1]*noisy_dt[i][0])) - noisy_dt[i][1])**2
    return d
### Nelder Mead    
from scipy import optimize
nelder_mead = optimize.minimize(rational, [0.2, -0.8], method='Nelder-Mead')

######### Exh search

a_list = np.arange(0.1,0.4, 0.001)
b_list = np.arange(-1.1,-0.6,0.001)
d_list = list()

def rational_func(a,b):
    d=0
    for i in range(len(noisy_dt)):
       d = d+((a/(1+b*noisy_dt[i][0])) - noisy_dt[i][1])**2
    return d

d_list = list()
def exh_search():
    for a in a_list:
        for b in b_list:
            d_list.append([rational_func(a,b),a,b])
    return (min(d_list))        
        
exh_sch = exh_search()        

exh_sch = [88.96912689956204, 0.40000000000000024, -0.7200000000000419]
        
####### Gauss
#a_list = np.arange(1,1.23, 0.001)
#b_list = np.arange(-0.17,0, 0.001)
def rational_func(a,b):
    d=0
    for i in range(len(noisy_dt)):
       d = d+((a/(1+b*noisy_dt[i][0])) - noisy_dt[i][1])**2
    return d
d_list = list()
list_a = list()
list_b = list()


def iteration_i():
    for a in a_list:
        list_a.append([rational_func(a,b_list[0]),a])
    a_opt=min(list_a)[1]
    for b in b_list:
        list_b.append([rational_func(a_opt,b),b])
    b_opt=min(list_b)[1]
    return a_opt, b_opt

#iteration_i()

def iteration_opt():
    for a in a_list:
        list_a.append([rational_func(a,b_opt),b_opt])
    a_opt=min(list_a)[1]
    for b in b_list:
        list_b.append([rational_func(a_opt,b),b])
    b_opt=min(list_b)[1]
    return a_opt, b_opt



def gauss():
    
    for i in range(0,100,1):
        if fabs(d_list[-1][0] - d_list[-2][0]) > 0.0001:
            if i ==0:
                a_opt,b_opt = iteration_i()
            else:
                a_opt,b_opt = iteration_opt()
            
            #i += 1
            d_list.append([rational_func(a_opt,b_opt),a_opt,b_opt])  
        elif fabs(d_list[-1][0] - d_list[-2][0]) < 0.0001:
            print(d_list[-1])
            break
       


gauss = gauss()  


d_list.append([rational_func(0.4,-0.72),0.4,-0.72])
d_list.append([rational_func(0.3,-0.8),0.3,-0.8])

gauss = [88.975423, 0.4005,-0.7200]
      
##################### Plotting ################3
predicted_lin = noisy_df.copy()
predicted_lin.index = predicted_lin.x
predicted_lin.index[0]
pred_mead = list()
pred_exh = list()
pred_gauss = list()



for i in range(len(noisy_dt)):
    pred_mead.append(F_rational(predicted_lin.index[i], nelder_mead['x'][0], nelder_mead['x'][1]))
for i in range(len(noisy_dt)):
    pred_exh.append(F_rational(predicted_lin.index[i], exh_sch[1], exh_sch[2]))
for i in range(len(noisy_dt)):
    pred_gauss.append(F_rational(predicted_lin.index[i], gauss[1], gauss[2]))
    

pred_mead = pd.Series(pred_mead)
pred_exh = pd.Series(pred_exh)
pred_gauss = pd.Series(pred_gauss)


predicted_lin['y_pred_mead'] = pred_mead.values
predicted_lin['y_pred_exh'] = pred_exh.values
predicted_lin['y_pred_gauss'] = pred_gauss.values


predicted_lin.drop('x', axis=1, inplace=True)

plt.scatter(noisy_df.x,noisy_df.y)
plt.plot([min(predicted_lin.index),max(predicted_lin.index)], 
          [min(predicted_lin.y_pred_gauss),max(predicted_lin.y_pred_gauss)], 'b', label='Gauss Regression line')
plt.plot([min(predicted_lin.index),max(predicted_lin.index)], 
          [min(predicted_lin.y_pred_mead),max(predicted_lin.y_pred_mead)], 'r', label='Nelder-Mead Regression line')
plt.plot([min(predicted_lin.index),max(predicted_lin.index)], 
          [min(predicted_lin.y_pred_exh),max(predicted_lin.y_pred_exh)], 'y', label='Exh Search Regression line')

plt.title('Regression line comparasion for Rational function')
plt.legend()
plt.show()

