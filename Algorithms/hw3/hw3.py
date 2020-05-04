# -*- coding: utf-8 -*-
"""
Lab 3

@author: Turukhanov Egor
"""

############# Second ###############
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
from scipy import optimize



alpha = random.uniform(0, 1)
beta = random.uniform(0, 1)

noisy_dt= list()
def noisy(al, bt, elements):
    for i in range(0, elements,1):
        x=i/100
        y=al*x+bt+np.random.normal(0,1)
        noisy_dt.append([x,y])
        
noisy(alpha, beta, 1000)

plt.scatter(noisy_df.x,noisy_df.y)
plt.title('Scatter Plot of Noisy Data')
plt.show()

def F_linear(x, a, b):
    return a*x+b

def lst_sqr(x, y, a, b):                                                                
    lin = (linear(x, a, b) - y)**2                                 
    return np.sum(lin) 

def lsq_lin(x):
    d=0
    for i in range(len(noisy_dt)):
       d = d+((x[0]*noisy_dt[i][0]+x[1]) - noisy_dt[i][1])**2
    return d
noisy_df = pd.DataFrame(noisy_dt, columns=['x', 'y'])
########## Gradient decend
noisy_df['x'].values

def gradient_descent(x, y):
    a = 0
    b = 0
    Δl = np.Infinity                                                                
    l = lst_sqr(x, y, a, b)
    δ = 0.001  # The learning Rate
    max_iterations = 100000  # The number of iterations to perform gradient descent
    i = 0
    L = 0.01
    n = float(len(noisy_dt)) # Number of elements in X
    while abs(Δl) > δ:
        i += 1

# Performing Gradient Descent 
    #for i in range(epochs): 
        #Y_pred = a*noisy_df['x'].values + b  # The current predicted value of Y
        D_a = np.sum(2*x*(a*x+b-y))/n  # Derivative wrt m
        D_b = np.sum(2*(a*x+b-y))/n# Derivative wrt c
        a = a - L * D_a  # Update a
        b = b - L * D_b  # Update b
        l_new = lst_sqr(x, y, a, b)                                                      
        Δl = l - l_new                                                           
        l = l_new
        print(i)
        print(l)
    print(a, b)
    return a,b 



%time a, b = gradient_descent(noisy_df.x.values, noisy_df.y.values)

print (a, b)
Y_pred = a*noisy_df['x'].values + b
plt.scatter(noisy_df['x'].values, noisy_df['y'].values) 
plt.plot([min(noisy_df['x'].values), max(noisy_df['x'].values)], [min(Y_pred), max(Y_pred)], color='red')  # regression line
plt.show()

a_gd = a
b_gd = b

### CG    ####
%time  cg = optimize.minimize(lsq_lin, [0 ,0], method='CG', options={'gtol': 0.001, 'maxiter': 1000, 'disp': True, 'return_all': True})
a_cg = cg['x'][0]
b_cg = cg['x'][1]

#### Newton#######
def linear(x, a, b):                                                        
    z = a*x + b                                              
    return z

#linear(noisy_df.x.values, 0.53,0.84)
     
def lst_sqr(x, y, a, b):                                                                
    lin = (linear(x, a, b) - y)**2                                 
    return np.sum(lin) 

#lst_sqr(noisy_df.x.values, noisy_df.y.values, 0.53, 0.84)

def gradient(x, y, a, b):
    grad_a = np.sum(2*x*(a*x+b-y))
    grad_b = np.sum(2*(a*x+b-y))
    return np.array([grad_a,grad_b])                   
                                              
#gradient(noisy_df.x.values, noisy_df.y.values, 0.53, 0.84)

def hessian(x, y, a, b):                                                          
    d1 = np.sum(2*x**2)                  
    d2 = np.sum(2)                  
    d3 = np.sum(2*x)                  
    H = np.array([[d1, d2],[d2, d3]])                                           
    return H

def newtons_method(x, y):                                                             
   
    # Initialize                                                                   
    a = 0                                                                     
    b = 0                                                               
    Δl = -np.Infinity                                                                
    l = lst_sqr(x, y, a, b)                                                                 
    # Convergence Conditions                                                        
    δ = 0.001                                                                 
    max_iterations = 1000                                                            
    i = 0                                                                           
    while abs(Δl) > δ and i < max_iterations:                                       
        i += 1                                                                      
        g = gradient(x, y, a, b)                                                      
        hess = hessian(x, y, a, b)                                                 
        H_inv = np.linalg.inv(hess)                                                 
        matr = np.dot(H_inv, g.T)                                                             
        Δa = matr[0]                                                             
        Δb = matr[1]                                                             
                                                                                    
        # Perform our update step                                                    
        a -= Δa                                                                 
        b -= Δb                                                                 
                                                                                    
        # Update the least_squares at each iteration                                     
        l_new = lst_sqr(x, y, a, b)                                                      
        Δl = l - l_new                                                           
        l = l_new
        print(l)
        print(i)
                                                                
    return np.array([a, b])                                 

%time a_nt, b_nt = newtons_method(noisy_df.x.values, noisy_df.y.values)


### Levenberg-Marquardt    
def least_sqrs_diff(args):
    a, b = args
    return ((a*noisy_df.x.values + b) - noisy_df.y.values)**2
x0=[1, 0]
    
%time least_squares = optimize.leastsq(least_sqrs_diff, x0, full_output=True,xtol=0.001)

a_lma = least_squares[0][0]
b_lma = least_squares[0][1]

############################# plotting ##############################

Y_pred_gd = a_gd*noisy_df['x'].values + b_gd
Y_pred_cg = a_cg*noisy_df['x'].values + b_cg
Y_pred_nt = a_nt*noisy_df['x'].values + b_nt
Y_pred_lma = a_lma*noisy_df['x'].values + b_lma

plt.scatter(noisy_df['x'].values, noisy_df['y'].values) 
plt.plot([min(noisy_df['x'].values), max(noisy_df['x'].values)], [min(Y_pred_lma), max(Y_pred_lma)], color='red', label='LMA Regression Line')  # regression line
plt.plot([min(noisy_df['x'].values), max(noisy_df['x'].values)], [min(Y_pred_gd), max(Y_pred_gd)], color='blue', label='GD Regression Line')  # regression line
plt.plot([min(noisy_df['x'].values), max(noisy_df['x'].values)], [min(Y_pred_cg), max(Y_pred_cg)], color='yellow', label='CG Regression Line')  # regression line
plt.plot([min(noisy_df['x'].values), max(noisy_df['x'].values)], [min(Y_pred_nt), max(Y_pred_nt)], color='green', label='Newton Regression Line')  # regression line
plt.title('Comparison of different unconstrained nonlinear optimisation algorithms for linear regression')
plt.legend()
plt.show()
######################## Rational ####################
'''def noisy(al, bt, elements):
    for i in range(0, elements,1):
        x=i/100
        y=al*x+bt+np.random.normal(0,1)
        noisy_dt.append([x,y])
        
noisy(alpha, beta, 1000)
''''

def rational(x, a, b):                                                        
    return a/(1+b*x)

def lst_sqr(x, y, a, b):                                                                
    lin = ((rational(x, a, b) - y)**2)                                 
    return np.sum(lin) 


# Performing Gradient Descent 
def gradient_descent(x, y):
    a = 1
    b = 0
    Δl = np.Infinity                                                                
    l = lst_sqr(x, y, a, b)
    δ = 0.0001  # The learning Rate
    max_iterations = 1000  # The number of iterations to perform gradient descent
    i = 0
    L = 0.001
    n = float(len(noisy_dt))
    b*x+1 != 0
    # Number of elements in X
    while abs(Δl) > δ:
       try:
           
           i += 1

    # Performing Gradient Descent 
           D_a = (-2/n)*np.sum((-a+b*x*y + y)/((b*x+1)**2))  # Partial Derivative of a
           D_b = (-2/n)*np.sum(a*x*(a-y*(b*x+1))/((b*x+1)**3))# Partial Derivative of b
           a = a - L * D_a  # Update a
           b = b - L * D_b  # Update b
           l_new = lst_sqr(x, y, a, b)                                                      
           Δl = l - l_new                                                           
           l = l_new
           print(l)
       except:
           pass
        #print(i)
    #print(l)
    print(i, l, a, b)
    return a,b 



%time a, b = gradient_descent(noisy_df.x.values, noisy_df.y.values)
print (a, b)
Y_pred = a/(1+noisy_df['x'].values*b)
plt.scatter(noisy_df['x'].values, noisy_df['y'].values) 
plt.plot(noisy_df['x'].values, Y_pred, color='red')  # regression line
plt.show()

a_gd_rat = a
b_gd_rat = b

### CG    ####
def rational(x):
    return np.sum(((x[0]/(1+x[1]*noisy_df.x.values)) - noisy_df.y.values)**2)


'''def rational(x):
    d = np.sum(((x[0]/(1+x[1]*noisy_dt[i][0])) - noisy_dt[i][1])**2)
    return d

'''
def jacob(x):
    el1 = np.sum(-2*(-x[0]+x[1]*noisy_df.x.values*noisy_df.y.values + noisy_df.y.values)/((x[1]*noisy_df.x.values+1)**2)) 
    el2 = np.sum(-2*x[0]*noisy_df.x.values*(x[0]-noisy_df.y.values*(x[1]*noisy_df.x.values+1))/((x[1]*noisy_df.x.values+1)**3))
    return np.array([el1,el2])

%time cg = optimize.minimize(rational, (0,0), method='CG', jac=jacob, options={'gtol': 0.001, 'maxiter': 100000, 'disp': True, 'return_all': True})
a_cg_rat = cg['x'][0]
b_cg_rat = cg['x'][1]
print(a_cg_rat, b_cg_rat)

#### Newton#######
def rational(x):
    d = np.sum(((x[0]/(1+x[1]*noisy_df.x.values)) - noisy_df.y.values)**2)
    return d

%time optimize.minimize(rational, (0,0), method='Newton-CG', jac=jacob, options={'gtol': 0.001, 'maxiter': 100000, 'disp': True, 'return_all': True})
%time nw = optimize.minimize(rational, (0,0), method='Newton-CG', jac=jacob, options={'maxiter': 100000, 'disp': True, 'return_all': True})
a_nt_rat = nw['x'][0]
b_nt_rat = nw['x'][1]

print(a_nt_rat, b_nt_rat)
### Levenberg-Marquardt    
def least_sqrs_diff(args):
    a, b = args
    return ((a/(1+noisy_df.x.values*b)) - noisy_df.y.values)**2
x0=[0, 0]
def jacas(args):
    a, b = args
    el1 = np.sum(-2*(-a+b*noisy_df.x.values*noisy_df.y.values + noisy_df.y.values)/((b*noisy_df.x.values+1)**2)) 
    el2 = np.sum(-2*a*noisy_df.x.values*(a-noisy_df.y.values*(b*noisy_df.x.values+1))/((b*noisy_df.x.values+1)**3))
    return np.array([el1,el2])
    

%time optimize.leastsq(least_sqrs_diff, x0)
%time least_squares = optimize.leastsq(least_sqrs_diff, x0,full_output=True,xtol=0.001)
a_lma_rat = least_squares[0][0]
b_lma_rat = least_squares[0][1]
print(a_lma_rat, b_lma_rat)

############################# plotting ##############################

Y_pred_gd = a_gd_rat/(1+noisy_df['x'].values*b_gd_rat)
Y_pred_cg = a_cg_rat/(1+noisy_df['x'].values*b_cg_rat)
Y_pred_nt = a_nt_rat/(1+noisy_df['x'].values*b_nt_rat)
Y_pred_lma = b_lma_rat/(1+noisy_df['x'].values*b_lma_rat)

plt.scatter(noisy_df['x'].values, noisy_df['y'].values) 
plt.plot([min(noisy_df['x'].values), max(noisy_df['x'].values)], [min(Y_pred_lma), max(Y_pred_lma)], color='red', label='LMA Regression Line')  # regression line
plt.plot([min(noisy_df['x'].values), max(noisy_df['x'].values)], [min(Y_pred_gd), max(Y_pred_gd)], color='blue', label='GD Regression Line')  # regression line
plt.plot([min(noisy_df['x'].values), max(noisy_df['x'].values)], [min(Y_pred_cg), max(Y_pred_cg)], color='yellow', label='CG Regression Line')  # regression line
plt.plot([min(noisy_df['x'].values), max(noisy_df['x'].values)], [min(Y_pred_nt), max(Y_pred_nt)], color='green', label='Newton Regression Line')  # regression line
plt.title('Comparison of different unconstrained nonlinear optimisation algorithms')
plt.legend()
plt.show()



Y_pred_cg = a_cg_rat/(1+noisy_df['x'].values*b_cg_rat)
Y_pred_gd = a_gd_rat/(1+noisy_df['x'].values*b_gd_rat)
Y_pred_nw = a_nt_rat/(1+noisy_df['x'].values*b_nt_rat)
Y_pred_lsq = a_lma_rat/(1+noisy_df['x'].values*b_lma_rat)
plt.scatter(noisy_df['x'].values, noisy_df['y'].values) 
plt.plot(noisy_df['x'].values, Y_pred, color='red', label='CD Regression Line')  # regression line
plt.plot(noisy_df['x'].values, Y_pred_gd, color='yellow', label='GG Regression Line')  # regression line
plt.plot(noisy_df['x'].values, Y_pred_nw, color='green', label='Newton Regression Line')  # regression line
plt.plot(noisy_df['x'].values, Y_pred_lsq, color='blue', label='LMA Regression Line')  # regression line
plt.title('Comparison of different unconstrained nonlinear optimisation algorithms for rational function')
plt.legend()
plt.show()


plt.plot(Y_pred_cg)
plt.plot(Y_pred_gd)
plt.plot(Y_pred_nw)
plt.plot(Y_pred_lsq)

##########################################################

def jacob(x):
    el1 = np.sum(-2*(-x[0]+x[1]*noisy_df.x.values*noisy_df.y.values + noisy_df.y.values)/((x[1]*noisy_df.x.values+1)**2)) 
    el2 = np.sum(-2*x[0]*noisy_df.x.values*(x[0]-noisy_df.y.values*(x[1]*noisy_df.x.values+1))/((x[1]*noisy_df.x.values+1)**3))
    return np.array([el1,el2])



def F_linear(x, a, b):
    return a*x+b
    
def F_rational(x, a, b):
    return a/(1+b*x)       

def lsq_lin(x):
    d=0
    for i in range(len(noisy_dt)):
       d = d+((x[0]*noisy_dt[i][0]+x[1]) - noisy_dt[i][1])**2
    return d

def rational(x, a, b):                                                        
    return a/(1+b*x)

#linear(noisy_df.x.values, 0.53,0.84)
     
def lst_sqr(x, y, a, b):                                                                
    lin = ((rational(x, a, b) - y)**2)                                 
    return np.sum(lin) 

#lst_sqr(noisy_df.x.values, noisy_df.y.values, 0.53, 0.84)

def gradient(x, y, a, b):
    grad_a = np.sum(-2*(-a+b*x*y + y)/((b*x+1)**2))
    grad_b = np.sum(-2*a*x*(a-y*(b*x+1))/((b*x+1)**3))
    return np.array([grad_a,grad_b])                   
                                              
#gradient(noisy_df.x.values, noisy_df.y.values, 0.53, 0.84)

def hessian(x, y, a, b):                                                          
    d1 = np.sum(2/(b*x-1))                  
    d2 = np.sum(2*a*x**3*(3*a-2*(b*x*y+y))/((b*x+1)**4))                  
    d3 = np.sum(-2*x*(-2*a+b*x*y+y)/((b*x+1)**4))                  
    H = np.array([[d1, d2],[d2, d3]])                                           
    return H

def newtons_method(x, y):                                                             
    """
    :param x (np.array(float)): Vector of Boston House Values in dollars
    :param y (np.array(boolean)): Vector of Bools indicting if house has > 2 bedrooms:
    :returns: np.array of logreg's parameters after convergence, [Θ_1, Θ_2]
    """

    # Initialize                                                                   
    a = 5                                                                     
    b = 5                                                               
    Δl = +np.Infinity                                                                
    l = lst_sqr(x, y, a, b)                                                                 
    # Convergence Conditions                                                        
    δ = 0.001                                                                 
    max_iterations = 1000                                                           
    i = 0                                                                           
    while abs(Δl) > δ and i < max_iterations:                                       
        i += 1                                                                      
        g = gradient(x, y, a, b)                                                      
        hess = hessian(x, y, a, b)                                                 
        H_inv = np.linalg.inv(hess)                                                 
        # @ is syntactic sugar for np.dot(H_inv, g.T)¹
        matr = np.dot(H_inv, g.T)                                                             
        Δa = matr[0]                                                             
        Δb = matr[1]                                                             
                                                                                    
        # Perform our update step                                                    
        a -= Δa                                                                 
        b -= Δb                                                                 
                                                                                    
        # Update the least_squares at each iteration                                     
        l_new = lst_sqr(x, y, a, b)                                                      
        Δl = l - l_new                                                           
        l = l_new
        print(l)                                                                
    return np.array([a, b])                                 


a_nt_rat, b_nt_rat = newtons_method(noisy_df.x.values, noisy_df.y.values)
