# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 14:51:36 2019

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

np.random.randint(1000, size=50)
np.random.rand(50)
vector = np.random.rand(100)

vectors = []
# Vectors Generation ####################
for i in range(50,  550, 50):
    vector = np.random.randint(1000, size=i)
    vectors.append(vector)


# Constant ######################
def contant(data):
    return(500)

total_time = []
vector_mean = pd.DataFrame(columns=['Time'])
for vector_i in vectors:
    for i in range(1, 5, 1):
        t = timeit.default_timer()
        contant(vector_i)
        elapsed_time = timeit.default_timer() - t
        total_time.append(elapsed_time)
    vector_mean = vector_mean.append({'Time': np.mean(total_time)}, ignore_index=True)
    
vector_mean.index = range(50,  550, 50)
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_ylabel('Time')
ax1.set_xlabel('Size of vector (N)')
ax1.set_title('Constant func for vector elements')
plt.plot(vector_mean.index, vector_mean.Time, 'ob')
plt.plot(vector_mean.index, vector_mean.Time, '--', label='Emperical')
plt.plot(vector_mean.index, np.repeat(np.mean(vector_mean.Time), 10), 'r', label='Theoretical')
plt.legend()
plt.show()

np.array(np.mean(vector_mean.Time))
np.repeat(np.mean(vector_mean.Time), 10)


# Sum ######################
total_time = []
vector_mean = pd.DataFrame(columns=['Time'])
for vector_i in vectors:
    for i in range(1, 5, 1):
        t = timeit.default_timer() 
        sum(vector_i)
        elapsed_time = timeit.default_timer() - t
        total_time.append(elapsed_time)
    vector_mean = vector_mean.append({'Time': np.mean(total_time)}, ignore_index=True)
    
vector_mean.index = range(50,  550, 50)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_ylabel('Time')
ax1.set_xlabel('Size of vector (N)')
ax1.set_title('Sum of vector elements')
plt.plot(vector_mean.index, vector_mean.Time, 'ob', label='Emperical')
plt.plot(vector_mean.index, vector_mean.Time, '--')
plt.plot([min(vector_mean.index),max(vector_mean.index)], [min(vector_mean.Time),
          max(vector_mean.Time)], 'r', label='Theoretical')
plt.legend()
plt.show()


# Geom ######################

def geom(data):
    geom=1
    for i in data:
        geom=data*i
    return(geom)

total_time = []
vector_mean = pd.DataFrame(columns=['Time'])
for vector_i in vectors:
    for i in range(1, 5, 1):
        t = timeit.default_timer()
        geom(vector_i)
        elapsed_time = timeit.default_timer() - t
        total_time.append(elapsed_time)
    vector_mean = vector_mean.append({'Time': np.mean(total_time)}, ignore_index=True)

    
vector_mean.index = range(50,  550, 50)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_ylabel('Time')
ax1.set_xlabel('Size of vector (N)')
ax1.set_title('Product of vector elements')
plt.plot(vector_mean.index, vector_mean.Time, 'ob', label='Emperical')
plt.plot(vector_mean.index, vector_mean.Time, '--')
plt.plot([min(vector_mean.index),max(vector_mean.index)], 
          [min(vector_mean.Time),max(vector_mean.Time)], 'r', label='Theoretical')
plt.legend()
plt.show()


# Euclidian norm ######################

total_time = []
vector_mean = pd.DataFrame(columns=['Time'])
for vector_i in vectors:
    for i in range(1, 5, 1):
        t = timeit.default_timer()
        np.sqrt(sum(vector_i**2))
        elapsed_time = timeit.default_timer() - t
        total_time.append(elapsed_time)
    vector_mean = vector_mean.append({'Time': np.mean(total_time)}, ignore_index=True)
    
vector_mean.index = range(50,  550, 50)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_ylabel('Time in millisec*100')
ax1.set_xlabel('Size of vector (N)')
ax1.set_title('Euclidean norm of vector elements')
plt.plot(vector_mean.index, vector_mean.Time, 'ob', label='Emperical')
plt.plot(vector_mean.index, vector_mean.Time, '--')
plt.plot([min(vector_mean.index),max(vector_mean.index)], 
          [min(vector_mean.Time),max(vector_mean.Time)], 'r', label='Theoretical')
plt.legend()
plt.show()

# Polynom ######################
def poly_naive(A, x):
    p = 0
    for i, a in enumerate(A):
        p += (x ** i) * a
    return p


total_time = []
vector_mean = pd.DataFrame(columns=['Time'])
for vector_i in vectors:
    for i in range(1, 5, 1):
        pol_sum = 0
        t = timeit.default_timer()
        poly_naive(vector_i, 1.5)
        elapsed_time = timeit.default_timer() - t
        total_time.append(elapsed_time)
    vector_mean = vector_mean.append({'Time': np.mean(total_time)}, ignore_index=True)
    
vector_mean.index = range(50,  550, 50)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_ylabel('Time in millisec*100')
ax1.set_xlabel('Size of vector (N)')
ax1.set_title('Polynomial direct of vector elements')
plt.plot(vector_mean.index, vector_mean.Time, 'ob', label='Emperical')
plt.plot(vector_mean.index, vector_mean.Time, '--')
plt.plot([min(vector_mean.index),max(vector_mean.index)], 
          [min(vector_mean.Time),max(vector_mean.Time)], 'r', label='Theoretical')
plt.legend()
plt.show()

###### Horner ########################
def poly_horner(A, x):
    p = A[-1]
    i = len(A) - 2
    while i >= 0:
        p = p * x + A[i]
        i -= 1
    return p        


total_time = []
vector_mean = pd.DataFrame(columns=['Time'])
for vector_i in vectors:
    for i in range(1, 5, 1):
        pol_sum = 0
        t = timeit.default_timer() 
        poly_horner(vector_i, 1.5)
        elapsed_time = timeit.default_timer() - t
        total_time.append(elapsed_time)
    vector_mean = vector_mean.append({'Time': np.mean(total_time)}, ignore_index=True)
    
vector_mean.index = range(50,  550, 50)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_ylabel('Time in millisec*100')
ax1.set_xlabel('Size of vector (N)')
ax1.set_title('Polynomial Horner of vector elements')
plt.plot(vector_mean.index, vector_mean.Time, 'ob', label='Emperical')
plt.plot(vector_mean.index, vector_mean.Time, '--')
plt.plot([min(vector_mean.index),max(vector_mean.index)], 
          [min(vector_mean.Time),max(vector_mean.Time)], 'r', label='Theoretical')
plt.legend()
plt.show()

######## Bubble sorting ##################
def bubbleSort(arr):
    n = len(arr)
 
    # Traverse through all array elements
    for i in range(n):
 
        # Last i elements are already in place
        for j in range(0, n-i-1):
 
            # traverse the array from 0 to n-i-1
            # Swap if the element found is greater
            # than the next element
            if arr[j] > arr[j+1] :
                arr[j], arr[j+1] = arr[j+1], arr[j]


total_time = []
vector_mean = pd.DataFrame(columns=['Time'])
for vector_i in vectors:
    for i in range(1, 5, 1):
        pol_sum = 0
        t = timeit.default_timer() 
        bubbleSort(vector_i)
        elapsed_time = timeit.default_timer() - t
        total_time.append(elapsed_time)
    vector_mean = vector_mean.append({'Time': np.mean(total_time)}, ignore_index=True)
    
vector_mean.index = range(50,  550, 50)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_ylabel('Time in millisec*100')
ax1.set_xlabel('Size of vector (N)')
ax1.set_title('Bubble sorting of vector elements')
plt.plot(vector_mean.index, vector_mean.Time, 'ob', label='Emperical')
plt.plot(vector_mean.index, vector_mean.Time, '--')
plt.plot(vector_mean.index, np.arange(min(np.sqrt(vector_mean.Time)), 
                                      max(np.sqrt(vector_mean.Time)),
          (max(np.sqrt(vector_mean.Time)))/11)**2, 'r', label='Theoretical')
plt.legend()
plt.show()

###### Matrix product ##########

np.random.rand(50,50)

matrixes = []

# Matrix Generation ####################
for i in range(50,  550, 50):

    matrix_A = np.random.randint(1000, size=(i, i))
    matrix_B = np.random.randint(1000, size=(i, i))
    matrixes.append([matrix_A, matrix_B])




total_time = []
vector_mean = pd.DataFrame(columns=['Time'])
for vector_i in matrixes:
    for i in range(1, 5, 1):
        pol_sum = 0
        t = timeit.default_timer()
        vector_i[0].dot(vector_i[1])
        elapsed_time = timeit.default_timer() - t
        total_time.append(elapsed_time)
    vector_mean = vector_mean.append({'Time': np.mean(total_time)}, ignore_index=True)
    
vector_mean.index = range(50,  550, 50)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_ylabel('Time in millisec')
ax1.set_xlabel('Size of vector (N)')
ax1.set_title('Matrix mult of vector elements')
plt.plot(vector_mean.index, vector_mean.Time, 'ob', label='Emperical')
plt.plot(vector_mean.index, vector_mean.Time, '--')
plt.plot(vector_mean.index, np.arange(min(np.cbrt(vector_mean.Time)), 
                                      max(np.cbrt(vector_mean.Time)),
          (max(np.sqrt(vector_mean.Time))-min(np.cbrt(vector_mean.Time)))/4.5)**3, 
    'r', label='Theoretical O(n^3)')
plt.legend()
plt.show()

np.arange(0.1,1.1,0.1)**3
