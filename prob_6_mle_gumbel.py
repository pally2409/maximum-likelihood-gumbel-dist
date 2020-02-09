#import the required libraries
from __future__ import division
import math
import matplotlib.pylab as plt
import sympy as sp
from numpy.linalg import inv
from time import sleep
import numpy as np
import matplotlib.pyplot as plt
import statistics

#method to generate gumbel distribution
def genGumbDist(a,b,n):
    return np.random.gumbel(a,b,n)

#method to estimate alpha and beta using newton-raphson that takes in the gumbel distribution, tolerance
def newtonraphson(dist, tolerance):

    #initialize the number of iterations 
    itr = 1

    #find the standard deviation and mean required for method of moment estimators
    s = statistics.stdev(dist)
    m = statistics.mean(dist)

    #find method of moment estimators to initialize alpha and beta
    a = m - 0.4501 * s
    b = 0.7977 * s
    
    #run an infinite loop to run it till it converges or reaches maximum number of iterations
    while True:

        n = len(dist)

        #find first order derivative of alpha, and beta
        a_fod = (1/b) * (n - sum((math.exp(-(x_i-a)/b) for x_i in dist)))
        b_fod = sum(((x_i - a)/b**2) for x_i in dist) - n/b - sum(((x_i - a)/b**2)*math.exp(-(x_i - a)/b) for x_i in dist)

        #find second order derivatives of alpha and beta
        a_sod = -(1/b**2)*sum(math.exp((-(x_i - a)/b)) for x_i in dist)
        b_sod = n/b**2 - (2/b**3) * sum((x_i - a) for x_i in dist) + (2/b**3)*sum((x_i - a) * math.exp(-(x_i - a)/b) for x_i in dist) - (1/b**4)*sum(((x_i - a)**2) * math.exp(-(x_i - a)/b) for x_i in dist)
        ab_sod = -(n/b**2) + (1/b**2) * sum(math.exp(-(x_i - a)/b) for x_i in dist) - (1/b**3)*sum((x_i - a) * math.exp(-(x_i - a)/b) for x_i in dist)
        
        #find the gradient
        f = np.matrix([[a_fod], [b_fod]], dtype = np.float64)
        
        #find the hessian matrix
        h = np.matrix([[a_sod, ab_sod], [ab_sod, b_sod]], dtype = np.float64)

        #inverse of hessian matrix
        h_inv = np.linalg.pinv(h)

        #get the old alpha and beta in aa matrix form
        theta_0 = np.matrix([[a], [b]], dtype=np.float64)

        #find new alpha and beta
        theta_1 = theta_0 - (h_inv * f) 

        #get new alpha and beta from the matrix
        a = theta_1[0,0] 
        b = theta_1[1,0]

        #find changes between old values and new values of alpha and beta
        t = (theta_0[0,0] - theta_1[0,0]) * (theta_0[0,0] - theta_1[0,0]) + (theta_0[1,0] - theta_1[1,0]) * (theta_0[1,0] - theta_1[1,0])
   
        #check for convergance by comparing with tolerance or for number of iterations
        if t <= tolerance or t >= 500:

            #if t is small enough, then newton raphson has converged, break
            break

        #increment the iterations
        itr = itr + 1
        
    #return alpha and beta
    return a,b

#initialize alpha and beta
a = 2.5
b = 4.0

#set the tolerance 
tolerance = 0.00000000001

#array of mean and standard deviation of alpha and beta values for n = 100, 1000, 10000 
a_mean = []
b_mean = []
a_stdev = []
b_stdev = []

#running this for n = 100, 1000, 10000
for i in range(3):

    #create an array of alpha values for newton raphson being run for a given n 10 times
    a_values = []

    #create an array of alpha values for newton raphson being run for a given n 10 times
    b_values = []

    #run a loop to run the newton raphson 10 times for a given n
    for j in range(10):

        #generate the gumbel distribution
        gumbel_distribution = genGumbDist(2.5, 4.0, 100*10**i)

        #get the alpha and beta approximations from newton raphson
        a, b = newtonraphson(gumbel_distribution, tolerance)

        #append it to the array we have created to keep track of alpha and beta for a given n
        a_values.append(a)
        b_values.append(b)

    #find the mean and standard deviation for alpha and beta
    a_mean.append(np.mean(a_values))
    b_mean.append(np.mean(b_values))
    a_stdev.append(np.std(a_values))
    b_stdev.append(np.std(b_values))

#print the values 
print("alpha:" , end =  ", ")
print(a_mean)
print("alpha:" , end =  ", ")
print(a_stdev)
print("beta:" , end =  ", ")
print(b_mean)
print("beta:" , end =  ", ")
print(b_stdev)
    






