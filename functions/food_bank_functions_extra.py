#  Food Bank Problem



import sys
import numpy as np
import plotly.express as px
import pandas as pd
import scipy.optimize as optimization








## Dynamic waterfilling algorithm that is more optimistic about town it is currently visiting than future towns
def waterfilling_optimistic(demands_predicted, demands_realized, b):
    n = np.size(demands_predicted)
    sorted_demands = np.sort(demands_predicted)
    allocations = np.zeros(n)
    bundle_remaining = b
    for i in range(n):
        sorted_demands = delete_sorted(sorted_demands, demands_predicted[i])
        new_sorted_list,index = insert_sorted((sorted_demands-(n-i)/n).clip(min=0),demands_realized[i])
        allocations[i] = (waterfilling_sorted(new_sorted_list, bundle_remaining))[index]
        bundle_remaining -= allocations[i]
    return allocations


# In[ ]:


## Dynamic waterfilling algorithm that is more pessimistic about town it is currently visiting than future towns
def waterfilling_pessimistic(demands_predicted, demands_realized, b):
    n = np.size(demands_predicted)
    sorted_demands = np.sort(demands_predicted)
    allocations = np.zeros(n)
    bundle_remaining = b
    for i in range(n):
        sorted_demands = delete_sorted(sorted_demands, demands_predicted[i])
        new_sorted_list,index = insert_sorted(sorted_demands+(n-i-1)/np.sqrt(n),demands_realized[i])
        allocations[i] = (waterfilling_sorted(new_sorted_list, bundle_remaining))[index]
        bundle_remaining -= allocations[i]
    return allocations


# In[13]:


## Online Water-filling algorithm where each agent is assigned infinite demand while finding optimal solution and allocation is readjusted
def waterfilling_online_3(demands_predicted, demands_realized, b):
    n = np.size(demands_predicted)
    allocations = np.zeros(n)
    bundle_remaining = b
    for i in range(n):
        future_allocations = waterfilling(np.append(np.Inf,demands_predicted[i+1:]), bundle_remaining)
        if future_allocations[0]>demands_realized[i] and i!=n-1:
            allocations[i] = demands_realized[i]
        else:
            allocations[i] = future_allocations[0]
        bundle_remaining -= allocations[i]
    return allocations

## O(n^2) version of online algorithm that needs waterfilling evaluated multiple times and budget over-estimated at first
def waterfilling_dynamic_budget_opt(demands_predicted, demands_realized, b, factor):
    n = np.size(demands_predicted)
    sorted_demands = np.sort(demands_predicted)
    allocations = np.zeros(n)
    bundle_remaining = b
    for i in range(n):
        sorted_demands = delete_sorted(sorted_demands, demands_predicted[i])
        new_sorted_list,index = insert_sorted(sorted_demands,demands_realized[i])
        if i<n-1:
            potential_alloc = min((waterfilling_sorted(new_sorted_list, bundle_remaining+factor*(n-i-1)/n))[index],demands_realized[i])
            if potential_alloc>=bundle_remaining:
                allocations[i] = min((waterfilling_sorted(new_sorted_list, bundle_remaining))[index],demands_realized[i])
            else:
                allocations[i] = potential_alloc
        else:
            allocations[i] = bundle_remaining
        bundle_remaining -= allocations[i]
    return allocations


## budget underestimated at first, and overestimated at end.
def waterfilling_weights_budget_adjust(weights, sorted_distribution, demands_realized, budget,factor_pess, factor_opt, turn):
    n = np.size(demands_realized)
    distribution_size = np.size(sorted_distribution)
    distribution_weighted = weights*sorted_distribution
    allocations = np.zeros(n)
    budget_remaining = budget
    turning_point = int(turn*n)
    for i in range(turning_point):
        new_sorted_list,index = insert_sorted(sorted_distribution,demands_realized[i])
        if i<n-1 :
            if factor_pess*(n-i-1)/n<budget_remaining:
                allocations[i] = min((waterfilling_sorted_weights(new_sorted_list, weights,budget_remaining-factor_pess*(n-i-1)/n,index,n-i-1))[index],demands_realized[i])
            else:
                allocations[i] = min((waterfilling_sorted_weights(new_sorted_list, weights,budget_remaining,index,n-i-1))[index],demands_realized[i])
        else:
            allocations[i] = budget_remaining
        budget_remaining -= allocations[i]

    for i in range(turning_point,n):
        new_sorted_list,index = insert_sorted(sorted_distribution,demands_realized[i])
        if i<n-1 :
            potential_alloc = min((waterfilling_sorted_weights(new_sorted_list, weights,budget_remaining+factor_opt*(n-i-1)/n,index,n-i-1))[index],demands_realized[i])
            if potential_alloc>=budget_remaining:
                allocations[i] = min((waterfilling_sorted(new_sorted_list, budget_remaining))[index],demands_realized[i])
            else:
                allocations[i] = potential_alloc
        else:
            allocations[i] = budget_remaining
        budget_remaining -= allocations[i]
    return allocations



## Online Water-filling taking minimum of realized demand and predetermeined allocation
def waterfilling_online_1(demands_predicted, demands_realized, b):
    n = np.size(demands_predicted)
    prior_allocations_assignment = waterfilling(demands_predicted,b)
    allocations = np.zeros(n)
    bundle_remaining = b
    for i in range(n):
        allocations[i] = min(prior_allocations_assignment[i], demands_realized[i]) if i!=n-1 else bundle_remaining
        bundle_remaining -= allocations[i]
    return allocations


# In[7]:


## Tests
assert list(waterfilling_online_1(np.zeros(0), np.zeros(0), 5)) == []
assert list(waterfilling_online_1(np.array([1,2,3,4]), np.array([5,5,5,5]), 10)) == [1,2,3,4]
assert list(waterfilling_online_1(np.array([3,1,4,2]), np.array([2,3,2.5,1]), 8)) == [2,1,2.5,2.5]


# In[8]:


## Online Water-filling algorithm where each agent solves waterfilling with realized current demand and expected following demands
def waterfilling_online_2(demands_predicted, demands_realized, b):
    n = np.size(demands_predicted)
    allocations = np.zeros(n)
    bundle_remaining = b
    for i in range(n):
        allocations[i] = waterfilling(np.append(demands_realized[i],demands_predicted[i+1:]), bundle_remaining)[0]
        bundle_remaining -= allocations[i]
    return allocations


# In[9]:



## Tests
assert list(waterfilling_online_2(np.zeros(0), np.zeros(0), 5)) == []
assert list(np.around(waterfilling_online_2(np.array([1,2,3,4]), np.array([5,5,5,5]), 11),2)) == [3,2.67,2.67,2.67]
assert list(waterfilling_online_2(np.array([4,5,3,6]), np.array([2,1,8,6]), 15)) == [2,1,6,6]
assert list(waterfilling_online_2(np.array([4,5,3,6]), np.array([9,10,2,1]), 15)) == [4,4,2,5]


## Tests
assert list(waterfilling_online_3(np.zeros(0), np.zeros(0), 5)) == []
assert list(np.around(waterfilling_online_3(np.array([1,2,3,4]), np.array([5,5,5,5]), 11),2)) == [3,2.67,2.67,2.67]
assert list(waterfilling_online_3(np.array([4,5,3,6]), np.array([2,1,8,6]), 15)) == [2,1,6,6]
assert list(waterfilling_online_3(np.array([4,5,3,6]), np.array([9,10,2,1]), 15)) == [4,4,2,5]
