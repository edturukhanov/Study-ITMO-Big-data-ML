# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 12:43:14 2019

@author: Egor Turukhanov
"""

import numpy as np
#from linear_program import LinearProgram, LPSolution
import pandas as pd
import matplotlib.pyplot as plt

def generate_tikz_plot(solution):
    """Generates a 3D Tikz LaTeX coordinate strings for the intermediate solutions."""
    for i, soln in enumerate(solution.intermediates):
        coordinate = r'\coordinate (xnew{}) at '.format(i)
        coordinate += r'({}, {}, {});'.format(*[coord for coord in soln.flat])
        print(coordinate)

    draw = r'\draw[red] (x) node[circle, fill, inner sep=1pt]{{}} '
    for i in range(len(solution.intermediates)):
        draw += r'-- (xnew{}) node[circle, fill, inner sep=1pt]{{}} '.format(i)
    draw += r';'
    print(draw)

"""A -- An n x m numpy matrix of constraint coefficients
b -- A 1 x m numpy row vector of constraint RHS values
c -- A 1 x n numpy row vector of objective function coefficients"""


A = np.matrix([[1, 0.1,0.08,0.001,0.002], [1, 0.2,0.1,0.005,0.005]])
b = np.array([[100,8,6,2,0.4],])
c = np.array([[-0.013, -0.008],])


A = A.reshape(5,2)


A.shape
b.shape
c.shape

start_point=np.array([33, 33])
n,m = A.shape

def karmarkar(start_point):
    D = np.diagflat(start_point)
    c_tilde = np.matmul(c, D)
    A_tilde = np.matmul(A, D)
    A_tildeT = A_tilde.transpose()
    AAT_inverse = np.linalg.pinv(np.matmul(A_tilde, A_tildeT))
    # matrix multiplication is associative
    P = np.identity(m) - np.matmul(np.matmul(A_tildeT, AAT_inverse), A_tilde)
    cp_tilde = np.matmul(c_tilde, P)
    k = -0.5 / np.amin(cp_tilde)
    x_tilde_new = np.ones((1, m), order='F') + k * cp_tilde
    return np.matmul(x_tilde_new, D)

solution = []

def solve(start_point, tolerance=1e-5, max_iterations=50, verbose=True):
    
    
    """Uses Karmarkar's Algorithm to solve a Linear Program.
    start_point     -- A starting point for Karmarkar's Algorithm. Must be a row vector.
    tolerance       -- The stopping tolerance of Karmarkar's Algorithm.
    max_iterations  -- The maximum number of iterations to run Karmarkar's Algorithm.
    verbose         -- List all intermediate values.
    """
    x = start_point
    #solution = LPSolution()
    for i in range(max_iterations):
        x_new = karmarkar(x)
        #if verbose:
            #print(x_new)

        dist = np.linalg.norm(x - x_new)
        x = x_new
        solution.append([x,i,dist])
        print(np.float64(np.tensordot(c, solution[-1][0], axes=((0,1),(0,1)))))
        #solution.append(x)
        if dist < tolerance:
            break

    solution.append([x,i,dist])
    #solution.iterations = i
    #solution.tolerance = dist
    #solution = solution

#    return solution

solve(start_point=np.array([33, 33]))

func_res = []
for i in range(len(solution)):
    func_res.append([i,np.float64(np.tensordot(c, solution[i][0], axes=((0,1),(0,1))))])
    
#np.tensordot(c, soversar, axes=((0,1),(0,1)))

res_df = pd.DataFrame(func_res,columns=['iteration','function'])
#iterations_1 = np.arange(0,12)

plt.title('Graph of iteration')
plt.plot(res_df.function) 
plt.scatter(res_df.iteration,res_df.function, c='red')
print(solution)


total_time[-1]



################### Complex #######
R=np.array([[1.1, 1.1, 1.1, 1.2, 1.2, 1.2],])
C=np.array([[0.3, 0.4, 0.48, 0.3, 0.4, 0.48],])
f=R-C
A=np.matrix([[1,0,0,1, 0, 0],[0, 1, 0, 0, 1, 0],[0, 0, 1, 0, 0, 1],[-1, -1, -1, 0, 0, 0],
[0, 0, 0, -1, -1, -1],[-0.7, 0.3, 0.3, 0, 0, 0],[-0.5, 0.5, -0.5, 0, 0, 0],[0.3, 0.3, -0.7, 0, 0, 0],
[0, 0, 0, 0.6, -0.4, -0.4],[0, 0, 0, 0.35, -0.65, 0.35],[0, 0, 0, -0.4, -0.4, 0.6]])
b=np.array([[6000, 10000, 12000, -10000, -10000, 0, 0, 0, 0, 0, 0]])
c = f



A = np.matrix([[1, 0.1,0.08,0.001,0.002], [1, 0.2,0.1,0.005,0.005],
               [1,0.15,0.11,0.003,0.007],[1,0,0.01,0.1,0.002],
               [1,0.4,0.1,0.15,0.008],[1,0,0,0,0]])

b = np.array([[100,8,6,2,0.4],])

c = np.array([[0.013,0.008, 0.01, 0.002, 0.005, 0.001]])


A = A.reshape(5,6)
b = b.reshape(1,5)
c = c.reshape(1,2)

A.shape
b.shape
c.shape

start_point=np.array([16.6,16.6,16.6,16.6,16.6,16.6])
n,m = A.shape


def karmarkar(x):

    D = np.diagflat(x)
    c_tilde = np.matmul(c, D)
    A_tilde = np.matmul(A, D)
    A_tildeT = A_tilde.transpose()
    AAT_inverse = np.linalg.inv(np.matmul(A_tilde, A_tildeT))
    # matrix multiplication is associative
    P = np.identity(m) + np.matmul(np.matmul(A_tildeT, AAT_inverse), A_tilde)
    cp_tilde = np.matmul(c_tilde, P)
    k = -0.5 / np.amin(cp_tilde)
    x_tilde_new = np.ones((1, m), order='F') + k * cp_tilde
    return np.matmul(x_tilde_new, D) 

solution = []


def solve(start_point, tolerance=1e-5, max_iterations=1000, verbose=True):
    
    
    """Uses Karmarkar's Algorithm to solve a Linear Program.
    start_point     -- A starting point for Karmarkar's Algorithm. Must be a row vector.
    tolerance       -- The stopping tolerance of Karmarkar's Algorithm.
    max_iterations  -- The maximum number of iterations to run Karmarkar's Algorithm.
    verbose         -- List all intermediate values.
    """
    x = start_point
    #solution = LPSolution()
    for i in range(max_iterations):
        x_new = karmarkar(x)
        #if verbose:
            #print(x_new)

        dist = np.linalg.norm(x - x_new)
        x = x_new
        solution.append([x,i,dist])
        #print(np.float64(np.tensordot(c, solution[-1][0], axes=((0,1),(0,1)))))

        #solution.append(x)
        if dist < tolerance:
            break

    solution.append([x,i,dist])
    
solve(start_point)

func_res = []
for i in range(len(solution)):
    func_res.append([i,np.float64(np.tensordot(c, solution[i][0], axes=((0,1),(0,1))))])
    
#np.tensordot(c, soversar, axes=((0,1),(0,1)))

res_df = pd.DataFrame(func_res,columns=['iteration','function'])
#iterations_1 = np.arange(0,12)

plt.title('Graph of iteration')
plt.plot(res_df.function) 
plt.scatter(res_df.iteration,res_df.function, c='red')

33.0897 + 15.8681 + 17.5842 + 14.9414 + 15.8056 + 3.95775e-06




from pulp import *

# Creates a list of the Ingredients
Ingredients = ['CHICKEN', 'BEEF', 'MUTTON', 'RICE', 'WHEAT', 'GEL']

# A dictionary of the costs of each of the Ingredients is created
costs = {'CHICKEN': 0.013, 
         'BEEF': 0.008, 
         'MUTTON': 0.010, 
         'RICE': 0.002, 
         'WHEAT': 0.005, 
         'GEL': 0.001}

# A dictionary of the protein percent in each of the Ingredients is created
proteinPercent = {'CHICKEN': 0.100, 
                  'BEEF': 0.200, 
                  'MUTTON': 0.150, 
                  'RICE': 0.000, 
                  'WHEAT': 0.040, 
                  'GEL': 0.000}

# A dictionary of the fat percent in each of the Ingredients is created
fatPercent = {'CHICKEN': 0.080, 
              'BEEF': 0.100, 
              'MUTTON': 0.110, 
              'RICE': 0.010, 
              'WHEAT': 0.010, 
              'GEL': 0.000}

# A dictionary of the fibre percent in each of the Ingredients is created
fibrePercent = {'CHICKEN': 0.001, 
                'BEEF': 0.005, 
                'MUTTON': 0.003, 
                'RICE': 0.100, 
                'WHEAT': 0.150, 
                'GEL': 0.000}

# A dictionary of the salt percent in each of the Ingredients is created
saltPercent = {'CHICKEN': 0.002, 
               'BEEF': 0.005, 
               'MUTTON': 0.007, 
               'RICE': 0.002, 
               'WHEAT': 0.008, 
               'GEL': 0.000}

# Create the 'prob' variable to contain the problem data
prob = LpProblem("The Whiskas Problem", LpMinimize)

# A dictionary called 'ingredient_vars' is created to contain the referenced Variables
ingredient_vars = LpVariable.dicts("Ingr",Ingredients,0)

# The objective function is added to 'prob' first
prob += lpSum([costs[i]*ingredient_vars[i] for i in Ingredients]), "Total Cost of Ingredients per can"

# The five constraints are added to 'prob'
prob += lpSum([ingredient_vars[i] for i in Ingredients]) == 100, "PercentagesSum"
prob += lpSum([proteinPercent[i] * ingredient_vars[i] for i in Ingredients]) >= 8.0, "ProteinRequirement"
prob += lpSum([fatPercent[i] * ingredient_vars[i] for i in Ingredients]) >= 6.0, "FatRequirement"
prob += lpSum([fibrePercent[i] * ingredient_vars[i] for i in Ingredients]) <= 2.0, "FibreRequirement"
prob += lpSum([saltPercent[i] * ingredient_vars[i] for i in Ingredients]) <= 0.4, "SaltRequirement"

# The problem data is written to an .lp file
prob.writeLP("WhiskasModel2.lp")

# The problem is solved using PuLP's choice of Solver
prob.solve()

# The status of the solution is printed to the screen
print("Status:", LpStatus[prob.status])

# Each of the variables is printed with it's resolved optimum value
for v in prob.variables():
    print (v.name, "=", v.varValue)

# The optimised objective function value is printed to the screen    
print("Total Cost of Ingredients per can = ", value(prob.objective))
    

############################# Solution ########################
A = np.matrix([[1, 0.1,0.08,0.001,0.002], [1, 0.2,0.1,0.005,0.005],
               [0,1,0,0,0],[0,0,1,0,0],
               [0,0,0,1,0],[0,0,0,0,1]])

b = np.array([[100,8,6,2,0.4],])
c = np.array([[-0.013,-0.008]])

A = A.reshape(5,6)

A = np.matrix([[1, 0.1,0.08,0.001,0.002], [1, 0.2,0.1,0.005,0.005]])
b = np.array([[100,8,6,2,0.4],])
c = np.array([[-0.013, -0.008],])


A = A.reshape(5,2)


A.shape
b.shape
c.shape

start_point=np.array([50,50])
n,m = A.shape


def karmarkar(start_point):

    D = np.diagflat(start_point)
    c_tilde = np.matmul(c, D)
    A_tilde = np.matmul(A, D)
    A_tildeT = A_tilde.transpose()
    AAT_inverse = np.linalg.pinv(np.matmul(A_tilde, A_tildeT))
    # matrix multiplication is associative
    P = np.identity(m) + np.matmul(np.matmul(A_tildeT, AAT_inverse), A_tilde)
    cp_tilde = np.matmul(c_tilde, P)
    k = -0.5 / np.amin(cp_tilde)
    x_tilde_new = np.ones((1, m), order='F') + k * cp_tilde
    return np.matmul(x_tilde_new, D) 

solution = []

vector_mean = pd.DataFrame(columns=['Time'])

def solve(start_point, tolerance=1e-3, max_iterations=1000, verbose=True):
    
  #  t = timeit.default_timer()

    """Uses Karmarkar's Algorithm to solve a Linear Program.
    start_point     -- A starting point for Karmarkar's Algorithm. Must be a row vector.
    tolerance       -- The stopping tolerance of Karmarkar's Algorithm.
    max_iterations  -- The maximum number of iterations to run Karmarkar's Algorithm.
    verbose         -- List all intermediate values.
    """
    x = start_point
    #solution = LPSolution()
    for i in range(max_iterations):
        x_new = karmarkar(x)
        #if verbose:
            #print(x_new)

        dist = np.linalg.norm(x - x_new)
        x = x_new
        solution.append([x,i,dist])
        #print(np.float64(np.tensordot(c, solution[-1][0], axes=((0,1),(0,1)))))

        #solution.append(x)
        if dist < tolerance:
            break

    solution.append([x,i,dist])
#    elapsed_time = timeit.default_timer() - t
 #   total_time.append(elapsed_time)
    
for i in range(0,10,1):
    t = timeit.default_timer()
    solve(start_point)
    elapsed_time = timeit.default_timer() - t
    total_time.append(elapsed_time)

vector_mean = vector_mean.append({'Time': np.mean(total_time)}, ignore_index=True)




import timeit

func_res = []
for i in range(len(solution)):
    func_res.append([i,np.float64(np.tensordot(c, solution[i][0], axes=((0,1),(0,1))))])
    
#np.tensordot(c, soversar, axes=((0,1),(0,1)))

res_df = pd.DataFrame(func_res,columns=['iteration','function'])
#iterations_1 = np.arange(0,12)

plt.title('Graph of iteration')
plt.plot(res_df.function) 
plt.scatter(res_df.iteration,res_df.function, c='red')

np.tensordot(c, [[38.147,61.9888]], axes=((0,1),(0,1)))



import timeit
total_time = []
t = timeit.default_timer()
elapsed_time = timeit.default_timer() - t
total_time.append(elapsed_time)

############################## Simplex ################################
from scipy.optimize import linprog
import numpy as np
import matplotlib.pyplot as plt
from time import time
import pandas as pd


c = [0.013,0.008]
A_ub = [[0.001, 0.005],[0.002,0.005],[-0.080,-0.1],[-0.1,-0.2]]
b_ub = [2,0.4,-6,-8]
A_eq = [[1,1]]
b_eq = [100]
total_time = []
vector_mean = pd.DataFrame(columns=['Time'])
for i in range(0,10,1): 
  t = timeit.default_timer()  
  res=linprog(c,A_ub,b_ub,A_eq,b_eq, bounds=(0,None))
  
  #Total_time = time() - start_time
  elapsed_time = timeit.default_timer() - t
  total_time.append(elapsed_time)
vector_mean = vector_mean.append({'Time': np.mean(total_time)}, ignore_index=True)
    
  
print("Total Time: %0.10f seconds." % Total_time)
print(res)
print("Function is ", res.fun,"\nX:", res.x)

######################### Ellipsoid ######################


