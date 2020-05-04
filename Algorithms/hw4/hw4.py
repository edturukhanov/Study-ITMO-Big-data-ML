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
from scipy import optimize

def func(x):
   return 1/(x**2-3*x+2)
random.normalvariate(0,1)
def noisy(elements):
    for i in range(0, elements):
        x=(3*i)/elements
        if func(x) <-100:
            y=-100+random.normalvariate(0,1)
            noisy_dt.append([x,y])
        elif func(x) >=-100 and func(x) <=100:
            y=func(x)+random.normalvariate(0,1)
            noisy_dt.append([x,y])
        elif func(x) >100:
            y=100+random.normalvariate(0,1)
            noisy_dt.append([x,y])
            

noisy_dt= list()
noisy(1000)
noisy_df = pd.DataFrame(noisy_dt, columns=['x', 'y'])

plt.scatter(noisy_df.x,noisy_df.y)
plt.title('Scatter Plot of Noisy Data')
plt.show()

def F(x):
    return (x[0]*noisy_df.x.values+x[1])/(noisy_df.x.values**2+x[2]*noisy_df.x.values+x[3])
    
def lstsqr(x):
    return np.sum((F(x)-noisy_df.y.values)**2)
  
### Nelder Mead    
%time nelder_mead = optimize.minimize(lstsqr, (0.5,0.5,0.5,0.5),method='Nelder-Mead', options={'maxiter':1000,'xtol':0.01})
a_nm = nelder_mead['x'][0]
b_nm = nelder_mead['x'][1]
c_nm = nelder_mead['x'][2]
d_nm = nelder_mead['x'][3]
it_nm =  nelder_mead['nit']
#114 ms 


def F(x):
    return (x[0]*noisy_df.x.values+x[1])/(noisy_df.x.values**2+x[2]*noisy_df.x.values+x[3])
    
def lstsqr(x):
    return np.sum((F(x)-noisy_df.y.values)**2)
  
### NELDER_MEAD    
%time nelder_mead = optimize.minimize(lstsqr, (0.5,0.5,0.5,0.5), method='Nelder-Mead', options={'maxiter':1000})
a_nm = nelder_mead['x'][0]
b_nm = nelder_mead['x'][1]
c_nm = nelder_mead['x'][2]
d_nm = nelder_mead['x'][3]
it_nm =  nelder_mead['nit']
nelder_mead['fun']

### LMA 

def least_sqrs_diff(args):
    a, b, c, d = args
    return (((a*noisy_df.x.values+b)/(noisy_df.x.values**2+c*noisy_df.x.values+d)) - noisy_df.y.values)**2



def lstsqr_lma(x):
    return (F(x)-noisy_df.y.values)**2

x0=[-0.5,1,-1,1]
    
%time least_squares = optimize.leastsq(lstsqr_lma, x0,full_output=True)

a_lma = least_squares[0][0]
b_lma = least_squares[0][1]
c_lma = least_squares[0][2]
d_lma = least_squares[0][3]
it_lma =  least_squares[2]['nfev']
#6.98 ms
    
def lstsqr_lma_sum(x):
    return sum(F(x)-noisy_df.y.values)**2

lstsqr_lma_sum(least_squares[0])

### particles Swap
import psopy
from psopy import _minimize_pso
x0 = np.random.uniform(-1.5, 1.5, (1000, 4))
def F(x):
    return (x[0]*noisy_df.x.values+x[1])/(noisy_df.x.values**2+x[2]*noisy_df.x.values+x[3])
    
def lstsqr(x):
    return np.sum((F(x)-noisy_df.y.values)**2)

lstsqr_ =  lambda x: np.apply_along_axis(lstsqr, 1, x)

%time swap_part = _minimize_pso(lstsqr_, x0,max_iter=1000)

a_sp = swap_part['x'][0]
b_sp = swap_part['x'][1]
c_sp = swap_part['x'][2]
d_sp = swap_part['x'][3]
it_sp =  swap_part['nit']
swap_part['fun']

############## Plotting ##########################
Y_pred_nm = (a_nm*noisy_df.x.values+b_nm)/(noisy_df.x.values**2+c_nm*noisy_df.x.values+d_nm)
Y_pred_lma = (a_lma*noisy_df.x.values+b_lma)/(noisy_df.x.values**2+c_lma*noisy_df.x.values+d_lma)
Y_pred_sp = (a_sp*noisy_df.x.values+b_sp)/(noisy_df.x.values**2+c_sp*noisy_df.x.values+d_sp)


plt.scatter(noisy_df['x'].values, noisy_df['y'].values) 
plt.plot(noisy_df['x'].values, Y_pred_nm, color='red', label='Nelder Mead Regression Line')  # regression line
plt.plot(noisy_df['x'].values, Y_pred_lma, color='yellow', label='LMA Regression Line')  # regression line
plt.plot(noisy_df['x'].values, Y_pred_sp, color='green', label='Swarm Particles Regression Line')  # regression line
#plt.plot(noisy_df['x'].values, Y_pred_lsq, color='blue', label='LMA Regression Line')  # regression line
plt.title('Comparison of different optimisation algorithms')
plt.legend()
plt.show()


