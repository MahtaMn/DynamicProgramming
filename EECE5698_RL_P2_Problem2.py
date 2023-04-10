#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from numpy import linalg 
import random


# In[ ]:


# Initialization

# control space
a1 = np.array([0,0,0,0])
a2 = np.array([1,0,0,0])
a3 = np.array([0,0,0,1])
a = np.array([a1, a2 , a3])

#a1 = np.array([0,0,0,0])
#a2 = np.array([1,0,0,0])
#a3 = np.array([0,1,0,0])
#a4 = np.array([0,0,1,0])
#a5 = np.array([0,0,0,1])
#a = np.array([a1, a2 , a3, a4, a5])

# state space
s = np.array([     [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1],     [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1],     [1, 0, 0, 0],[1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1],     [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1]])   

# connectivity matrix
C = np.array([[0, 0, -1, 0], [1, 0, -1, -1], [0, 1, 0, 0], [-1, 1, 1, 0]])

gamma = 0.95
theta = 0.01

p0 = 0.05

# initial pi
policy0 = np.array([a1, a1, a1, a1, a1, a1, a1, a1, a1, a1, a1, a1, a1, a1, a1, a1])
no_control = np.array([     [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],     [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],     [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],     [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])   


# In[ ]:


# Creating Transition Matrices

M = {}        
for ii in range(len(a)):
    
    mat = np.zeros([s.shape[0], s.shape[0]])
    
    for i in range(s.shape[0]):
        for j in range(s.shape[0]):
            
            x = np.matmul(C, s[i])
            x = np.where(x > 0, 1, 0)
    
            xx = np.sum(np.absolute((s[j] - (x^a[ii]))))
        
            mat[i][j] = np.power(p0, xx) * np.power(1-p0, 4 - xx)
        
    M[ii] = mat 


# In[ ]:


# Creating Reward Matrices

Rsa = {}
Rsas = np.zeros([s.shape[0], s.shape[0]])

for ii in range(len(a)):
    
    for i in range(len(s)):
        for j in range(len(s)):
        
            Rsas[i][j] = 5*np.sum(s[j]) - np.sum(a[ii])
        
    Rsa[ii] = np.matmul(np.multiply(M[ii], Rsas), np.ones([s.shape[0], 1]))  


# In[ ]:


def policy_evaluation_matrixform(pi):
     
    #v = V.copy()
    V = np.empty([s.shape[0], 1])
    Rpi = np.empty((len(s), 1))
    Mpi = np.empty((s.shape[0], s.shape[0]))
    
    for i in range(len(s)):
        
        for ii in range(len(a)):
            if np.all(pi[i] == a[ii]):
                ind = ii
        
        Rpi[i] = Rsa[ind][i]
        Mpi[i] = M[ind][i]
    
    V = np.matmul(linalg.inv(np.eye(len(s)) - gamma*Mpi), Rpi)
    
    return V


# In[ ]:


def policy_iteration_matrixform(policy):
    
    it = 0
    L = np.empty((s.shape[0], len(a)))
    V = np.empty([s.shape[0], 1])
    
    while True:
        
        V = policy_evaluation_matrixform(policy)
       
        policy_stable = False
    
        old_policy = policy.copy()
        #print('old policy: ', old_policy)
        
        for i in range(len(a)):
            L[:,i] = (Rsa[i] + gamma*np.matmul(M[i], V)).ravel()
        
        indd = np.argmax(L, axis=1)
        
        for i in range(len(indd)):
            for ii in range(len(a)):
                
                if indd[i] == ii:
                    policy[i] = a[ii]
        #print('new policy: ', policy)
        
        if np.array_equiv(policy, old_policy):
            policy_stable = True
        #print('policy_stable: ', policy_stable)
        
        it += 1
        
        if policy_stable:
            break
            
    print('policy evaluated in iteration', it)     
    return V, policy


# In[ ]:


def value_iteration_matrixform():
    
    V = np.zeros([s.shape[0], 1])
    it = 0
    pi = np.empty((s.shape[0], s.shape[1]), dtype=int)
    indd = np.zeros([s.shape[0], 1])
    
    while True:
                  
        v = V.copy()
        
        L = np.empty((s.shape[0], len(a)))
        for i in range(len(a)):
            L[:,i] = (Rsa[i] + gamma*np.matmul(M[i], V)).ravel()
                         
        #for i in range(L.shape[0]):
        #    V[i] = np.max(L[i])
        
        V = np.amax(L, axis=1).reshape([16,1])
        #print('V: ',V)
        
        delta = np.max(np.absolute(v - V))
        #print('delta', delta)
        
        it += 1
        
        if delta < theta:
            break
    
    L = np.empty((s.shape[0], len(a)))
    for i in range(len(a)):
        L[:,i] = (Rsa[i] + gamma*np.matmul(M[i], V)).ravel()
    
    
    indd = np.argmax(L, axis=1).reshape([16,1])
    print(indd)
    
    for i in range(len(indd)):
        for ii in range(len(a)):
            if indd[i] == ii:
                pi[i] = a[ii]
            
    print('V computed in iteration', it)    
    return V, pi


# In[ ]:


# Call Policy Iteration   
V_optimal, pi_optimal = policy_iteration_matrixform(policy0) 
#print('optimal policy', pi_optimal)
#print('optimal V', V_optimal)


# In[ ]:


# Call Values Iteration
#V_optimal, pi_optimal = value_iteration_matrixform()
#print('optimal policy', pi_optimal)
#print('optimal V', V_optimal)


# In[ ]:


def average_activation(pi, num_traj, num_eps):
     
    D = {}
    A = []
    for i in range(num_eps):
        
        summ = 0
        tmp = []
        
        sk = np.empty((num_traj, s.shape[1]), dtype=int)
        sk[0] = random.choice(s)
        
        for ii in range(1, num_traj):
        
            x = np.matmul(C, sk[ii-1])
            x = np.where(x > 0, 1, 0)
            
            ind = np.where(s == sk[ii-1])[0][0]
            ak_1 = pi[ind]
            
            nk = np.random.binomial(size=4, n=1, p=p0)
    
            xx = x ^ ak_1
            sk[ii] = xx ^ nk 
        
            tmp.append((list(sk[ii-1]), list(ak_1)))
            summ += sum(sk[ii-1])
            
        D[i] = tmp
            
        A.append(summ/num_traj)
        
    avgA = sum(A)/num_eps    
    
    return avgA, D


# In[ ]:


num_traj = 200
num_eps = 100

avgA, D = average_activation(pi_optimal, num_traj, num_eps)
print(avgA)
avgA, D = average_activation(no_control, num_traj, num_eps)
print(avgA)


# In[ ]:




