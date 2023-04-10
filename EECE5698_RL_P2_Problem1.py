#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import random
"""
Visualization code to display found State Values and the optimum policy on the maze. 
"""
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# In[ ]:


# Initialization 
        
# state matrix
# w: wall - e: empty - b: bump - o: oil - g: goal - s: start
S = np.array([['w','w','w','w','w','w','w','w','w','w','w','w','w','w','w','w','w','w','w','w'],
             ['w','e','e','e','e','e','e','e','e','e','e','b','b','e','e','e','e','e','e','w'],
             ['w','b','b','b','e','w','e','e','o','e','e','e','e','e','e','e','o','e','e','w'],
             ['w','e','e','e','e','w','e','e','e','e','e','e','e','g','e','e','e','e','e','w'],
             ['w','e','o','w','w','w','w','w','w','w','w','w','w','w','w','w','w','e','e','w'], 
             ['w','b','e','w','e','e','o','e','e','b','e','e','e','e','e','e','e','b','e','w'],
             ['w','e','e','w','e','e','w','e','e','w','e','e','e','e','e','w','e','b','e','w'],
             ['w','e','b','w','e','e','w','e','e','w','b','b','w','w','w','w','e','b','e','w'],         
             ['w','e','e','e','e','e','w','e','e','w','e','e','e','e','e','w','e','b','e','w'],       
             ['w','e','e','e','e','e','w','e','e','w','e','e','e','e','e','w','e','e','e','w'],       
             ['w','w','w','w','w','e','w','e','e','w','w','e','e','e','e','w','e','e','o','w'], 
             ['w','e','e','e','e','e','w','e','e','e','w','e','e','w','e','w','w','w','e','w'],
             ['w','e','e','w','w','w','w','w','e','e','w','b','b','w','e','e','e','w','e','w'],
             ['w','e','e','e','e','e','e','w','e','e','w','e','e','w','e','e','e','w','e','w'], 
             ['w','b','b','e','e','e','e','w','e','e','w','e','e','w','e','e','e','e','e','w'],
             ['w','e','e','e','s','e','e','w','e','e','o','e','e','w','w','w','w','b','b','w'], 
             ['w','e','e','e','e','e','e','b','e','e','o','e','e','e','e','e','e','e','e','w'],
             ['w','w','e','e','e','e','e','w','w','w','w','w','w','e','o','e','e','o','e','w'], 
             ['w','e','e','e','e','e','e','o','e','e','e','e','e','e','e','e','e','e','e','w'],
             ['w','w','w','w','w','w','w','w','w','w','w','w','w','w','w','w','w','w','w','w']]) 


A = ['u', 'd', 'l', 'r'] # actions

gamma = 0.55
theta = 0.01
p0 = 0.02

# Initialize V and pi

V0 = np.zeros((S.shape[0], S.shape[1])) # all-zero initial state value
policy0 = np.empty((S.shape[0], S.shape[1]), dtype = str)

for i in range(policy0.shape[0]):
    for j in range(policy0.shape[1]):
        policy0[i][j] =  'l'      
        
max_iters = 1000       


# In[ ]:


def policy_evaluation(V, policy):
    
    it = 0
    
    while True:
        
        delta = 0
    
        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                
                if S[i][j] == 'w':
                    continue
                
                v = V[i][j].copy()
                
                action = policy[i][j]
                
                next_states, rewards, probs = neighbours(i, j, action)
                 
                summ = 0
                for ii, state in enumerate(next_states):        
                    summ += probs[ii] * (rewards[ii] + (gamma*V[state])) 
                 
                V[i][j] = summ
                delta = max(delta, abs(v - V[i][j]))
                #print('delta', delta)
                
        it += 1
        if delta < theta:
            break
      
    
    return V


# In[ ]:


def policy_iteration(V, policy):
    
    it = 0
    
    while True:
        
        # policy evaluation
        V = policy_evaluation(V, policy)
        #print('V: ', V)
        
        
        # policy improvement
        policy_stable = True
    
        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                
                if S[i][j] == 'w':
                    continue
                
                action = policy[i][j]
                
                v = V[i][j].copy()  
                    
                L = one_step_lookahead((i,j), V)
                    
                policy[i][j] = A[np.argmax(L)]
                
                if policy[i][j] != action:
                    policy_stable = False
                    
        #print('policy', policy)
        it += 1
        if policy_stable:
            break
            
            
    print('policy evaluated in iteration', it)
    return V, policy


# In[ ]:


def value_iteration(V):
    
    pi = policy0.copy()
    it = 0
    
    while True:
        
        delta = 0
        
        for i in range(S.shape[0]):
            for j in range(S.shape[1]):
                
                if S[i][j] == 'w':
                    continue
                
                v = V[i][j].copy()  
                    
                L = one_step_lookahead((i,j), V)
                               
                V[i][j] = np.max(L)
                
                delta = max(delta, abs(v - V[i][j]))    
  
        it += 1
        if delta < theta:
            break
        
        
    for i in range(1, S.shape[0]-1):
        for j in range(1, S.shape[1]-1):
            
            L = one_step_lookahead((i,j), V)
            
            pi[i][j] = A[np.argmax(L)]
           
            
    print('policy evaluated in iteration', it)        
    return V, pi


# In[ ]:


def one_step_lookahead(curr_state, V):
    
    L = [0, 0, 0, 0]
    
    for i, action in enumerate(A):
        
        next_states, rewards, probs = neighbours(curr_state[0], curr_state[1], action)
                        
        for ii, state in enumerate(next_states):        
            L[i] += probs[ii] * (rewards[ii] + (gamma*V[state])) 
            
     
    return L


# In[ ]:


def neighbours(curr_r, curr_c, selected_action):
    
    curr_state = (curr_r, curr_c)
    next_states = []
    probs = []
    
    for action in A:
        if action == selected_action:
            probs.append(1-p0)
        else:
            probs.append(p0/3)
            
        if action == 'u':
            next_r = curr_r- 1
            next_c = curr_c
        
        if action == 'd':
            next_r = curr_r + 1
            next_c = curr_c
        
        if action == 'l':
            next_r = curr_r
            next_c = curr_c - 1
        
        if action == 'r':
            next_r = curr_r
            next_c = curr_c + 1
                 
        next_states.append((next_r, next_c))
    
    #idx = A.index(selected_action)
    #if idx == 0 :
    #    probs = [1-p0, p0/3, p0/3, p0/3]
    #elif idx == 1:    
    #    probs = [p0/3, 1-p0, p0/3, p0/3]
    #elif idx == 2:
    #    probs = [p0/3, p0/3, 1-p0, p0/3]
    #elif idx == 3:
    #    probs = [p0/3, p0/3, p0/3, 1-p0]
   
    
    # look at next state's condition
    # return reward based on condition
    rewards = []
    r = 0
    
    for i, state in enumerate(next_states):
        
        if S[state] == 'w': # agent is hit by wall 
            
            next_states[i] = curr_state # stays in previous location
        
            if S[curr_state] == 'e':
                r = -1
            if S[curr_state] == 'b':
                r = -1 - 10
            if S[curr_state] == 'o':
                r = -1 - 5
            if S[curr_state] == 'g':
                r = 200     
            if S[curr_state] == 's':
                r = -1
                
        else:
            if S[state] == 'e':
                r = -1
            if S[state] == 'b':
                r = -1 - 10
            if S[state] == 'o':
                r = -1 - 5
            if S[state] == 'g':
                r = 200 - 1     
            if S[state] == 's':
                r = -1 
                
        rewards.append(r) 
        
             
    return next_states, rewards, probs


# In[ ]:


def get_nextstate(curr_r, curr_c, action):
    
    if action == 'u':
        r = curr_r - 1
        c = curr_c
        
    if action == 'd':
        r = curr_r + 1
        c = curr_c
        
    if action == 'l':
        r = curr_r 
        c = curr_c - 1
        
    if action == 'r':
        r = curr_r
        c = curr_c + 1
                 
    next_state = (r, c)

    
    return next_state


# In[ ]:


def get_path(pi):
    
    path = []
    
    startstate_r = np.where(S == 's')[0][0]
    startstate_c = np.where(S == 's')[1][0]
    
    goalstate_r = np.where(S == 'g')[0][0]
    goalstate_c = np.where(S == 'g')[1][0]
    
    curr_r = startstate_r
    curr_c = startstate_c
    
    path.append(tuple([(curr_r, curr_c) , pi[curr_r][curr_c]]))

    while True:
              
        next_state = get_nextstate(curr_r, curr_c, pi[curr_r][curr_c]) # this is a tuple
        direction = pi[next_state]
        
        path.append((next_state , direction))
    
        curr_r = next_state[0]
        curr_c = next_state[1]
        
        if curr_r == goalstate_r and curr_c == goalstate_c:
            break
            
            
    return path


# In[ ]:


# Call Policy Iteration   
#V_optimal, pi_optimal = policy_iteration(V0, policy0) 
# Call Values Iteration
V_optimal, pi_optimal = value_iteration(V0)
#pi_optimal
path = get_path(pi_optimal)
#path


# In[ ]:


""" 
Define and Visualize State Matrix 
See https://seaborn.pydata.org/generated/seaborn.heatmap.html for more info on arguments
"""
# This is a random matrix for example purposes. 
# Matrix is defined as 20x20 instead of 18x18 stated in the project description in order to treat borders as wall states
State_Matrix =     np.array([[255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
              [255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255],
              [255,   0,   0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255],
              [255,   0,   0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255],
              [255,   0,   0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,   0,   0, 255], 
              [255,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255],
              [255,   0,   0, 255,   0,   0, 255,   0,   0, 255,   0,   0,   0,   0,   0, 255,   0,   0,   0, 255],
              [255,   0,   0, 255,   0,   0, 255,   0,   0, 255,   0,   0, 255, 255, 255, 255,   0,   0,   0, 255],  
              [255,   0,   0,   0,   0,   0, 255,   0,   0, 255,   0,   0,   0,   0,   0, 255,   0,   0,   0, 255],  
              [255,   0,   0,   0,   0,   0, 255,   0,   0, 255,   0,   0,   0,   0,   0, 255,   0,   0,   0, 255], 
              
              [255, 255, 255, 255, 255,   0, 255,   0,   0, 255, 255,   0,   0,   0,   0, 255,   0,   0,   0, 255],
              
              [255,   0,   0,   0,   0,   0, 255,   0,   0,   0, 255,   0,   0, 255,   0, 255, 255, 255,   0, 255], 
              
              [255,   0,   0, 255, 255, 255, 255, 255,   0,   0, 255,   0,   0, 255,   0,   0,   0, 255,   0, 255], 
              [255,   0,   0,   0,   0,   0,   0, 255,   0,   0, 255,   0,   0, 255,   0,   0,   0, 255,   0, 255], 
              [255,   0,   0,   0,   0,   0,   0, 255,   0,   0, 255,   0,   0, 255,   0,   0,   0,   0,   0, 255],
              [255,   0,   0,   0,   0,   0,   0, 255,   0,   0,   0,   0,   0, 255, 255, 255, 255,   0,   0, 255],
              [255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255],
              [255, 255, 255,   0,   0,   0,   0, 255, 255, 255, 255, 255, 255,   0,   0,   0,   0,   0,   0, 255],
              [255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255],
              [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]])
        

plt.subplots(figsize=(10,7.5))
heatmap = sns.heatmap(State_Matrix, fmt=".2f", linewidths=0.25, linecolor='black',
                      cbar= False, cmap= 'rocket_r')
heatmap.set_facecolor('black') # Color for the NaN cells in the state matrix
plt.title('Maze Problem')
plt.show()


# In[ ]:


""" Function to always color the oil, bump, start, and green blocks.
 States are in the form of a list of (i,j) coordinates on the state matrix"""
def coloring_blocks(heatmap, oil_states, bump_states, start_state, end_state):
    # Adding red oil blocks
    for i in range(len(oil_states)):
        heatmap.add_patch(Rectangle((oil_states[i][1], oil_states[i][0]), 1, 1,
                                    fill=True, facecolor='red', edgecolor='red', lw=0.25))
    # Adding salmon bump blocks
    for i in range(len(bump_states)):
        heatmap.add_patch(Rectangle((bump_states[i][1], bump_states[i][0]), 1, 1,
                                    fill=True, facecolor='lightsalmon', edgecolor='lightsalmon', lw=0.25))
    # Adding start block (Blue)
    heatmap.add_patch(Rectangle((start_state[1], start_state[0]), 1, 1,
                                fill=True, facecolor='lightblue', edgecolor='lightblue', lw=0.25))

    # Adding end block (Green)
    heatmap.add_patch(Rectangle((end_state[1], end_state[0]), 1, 1,
                                fill=True, facecolor='lightgreen', edgecolor='lightgreen', lw=0.25))

# Example Use
plt.subplots(figsize=(10,7.5))    
heatmap = sns.heatmap(State_Matrix, fmt=".2f", linewidths=0.25, linecolor='black',
                      cbar= False, cmap= 'rocket_r')
heatmap.set_facecolor('black') # Color for the NaN cells in the state matrix
coloring_blocks(heatmap, oil_states=[(2,8),(2,16),(4,2),(5,6),(10,18),(15,10),(16,10),(17,14),(17,17),(18,7)],                 bump_states=[(1,11),(1,12),(2,1),(2,2),(2,3),(5,1),(5,9),(5,17),(6,17),(7,2),(7,10),(7,11),(7,17),                              (8,17),(12,11),(12,12),(14,1),(14,2),(15,17),(15,18),(16,7)],                 start_state=(15,4), end_state=(3,13))
plt.show()


# In[ ]:


# plot the value function values on the heat map
plt.subplots(figsize=(15,7.5))

# Create a 2D matrix of zeros with size of 20 x 20 
State_Matrix_heatmap = np.zeros((20,20)) 

# Assume V_s is a 18x18 matrix with calculated state values. 
# For example purposes, it is defined as a random matrix here.
V_s = V_optimal

for i in range(18):
    for j in range(18):
        # Assign new 2D matrix with the value function value at the current state
        State_Matrix_heatmap[i+1, j+1] = V_s[i][j]

# Plot the new heatmap of the new value function values with the original state and coloring blocks
heatmap = sns.heatmap(State_Matrix, fmt=".2f", annot= State_Matrix_heatmap, linewidths=0.25, linecolor='black',
                      cbar= False, cmap= 'rocket_r')

heatmap.set_facecolor('black') # Color for the NA cells in the state matrix
coloring_blocks(heatmap, oil_states=[(2,8),(2,16),(4,2),(5,6),(10,18),(15,10),(16,10),(17,14),(17,17),(18,7)],                 bump_states=[(1,11),(1,12),(2,1),(2,2),(2,3),(5,1),(5,9),(5,17),(6,17),(7,2),(7,10),(7,11),(7,17),                              (8,17),(12,11),(12,12),(14,1),(14,2),(15,17),(15,18),(16,7)],                 start_state=(15,4), end_state=(3,13))
plt.show()


# In[ ]:


# Define heatmap first
plt.subplots(figsize=(10, 7.5))
heatmap = sns.heatmap(State_Matrix, fmt=".2f", linewidths=0.25, linecolor='black', cbar=False, cmap='rocket_r')
heatmap.set_facecolor('black') 
coloring_blocks(heatmap, oil_states=[(2,8),(2,16),(4,2),(5,6),(10,18),(15,10),(16,10),(17,14),(17,17),(18,7)],                 bump_states=[(1,11),(1,12),(2,1),(2,2),(2,3),(5,1),(5,9),(5,17),(6,17),(7,2),(7,10),(7,11),(7,17),                              (8,17),(12,11),(12,12),(14,1),(14,2),(15,17),(15,18),(16,7)],                 start_state=(15,4), end_state=(3,13))
    

# Plot the route from the start state to the end state.
# This is just an example, you may want to keep pi* coordinates and actions in a different way
#route = pi_optimal

for i in range(1, pi_optimal.shape[0]-1):
    for j in range(1, pi_optimal.shape[1]-1):
        r = i # x_coordinate
        c = j # y_coordinate
        direction = pi_optimal[i][j]
        if direction == 'r':
            plt.arrow(c + 0.5, r + 0.5, 0.2, 0, width=0.06, color='black') # Right
        if direction == 'l':
            plt.arrow(c + 0.5, r + 0.5, -0.2, 0, width=0.06, color='black') # Left
        if direction == 'u':
            plt.arrow(c + 0.5, r + 0.5, 0, -0.2, width=0.06, color='black') # Up
        if direction == 'd':
            plt.arrow(c + 0.5, r + 0.5, 0, 0.2, width=0.06, color='black') # Down

# Show plot
plt.show()


# In[ ]:


# Define heatmap first
plt.subplots(figsize=(10, 7.5))
heatmap = sns.heatmap(State_Matrix, fmt=".2f", linewidths=0.25, linecolor='black', cbar=False, cmap='rocket_r')
heatmap.set_facecolor('black') 
coloring_blocks(heatmap, oil_states=[(2,8),(2,16),(4,2),(5,6),(10,18),(15,10),(16,10),(17,14),(17,17),(18,7)],                 bump_states=[(1,11),(1,12),(2,1),(2,2),(2,3),(5,1),(5,9),(5,17),(6,17),(7,2),(7,10),(7,11),(7,17),                              (8,17),(12,11),(12,12),(14,1),(14,2),(15,17),(15,18),(16,7)],                 start_state=(15,4), end_state=(3,13))
    

# Plot the route from the start state to the end state.
# This is just an example, you may want to keep pi* coordinates and actions in a different way
route = path

for state_cr, direction in route[:-1]:
    r = state_cr[0] # x_coordinate
    c = state_cr[1] # y_coordinate

    if direction == 'r':
        plt.arrow(c + 0.5, r + 0.5, 0.8, 0, width=0.04, color='black') # Right
    if direction == 'l':
        plt.arrow(c + 0.5, r + 0.5, -0.8, 0, width=0.04, color='black') # Left
    if direction == 'u':
        plt.arrow(c + 0.5, r + 0.5, 0, -0.8, width=0.04, color='black') # Up
    if direction == 'd':
        plt.arrow(c + 0.5, r + 0.5, 0, 0.8, width=0.04, color='black') # Down

# Show plot
plt.show()

