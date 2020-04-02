
# coding: utf-8

# # Food Bank Problem

# In[2]:


import sys
import numpy as np
import plotly.express as px
import pandas as pd
import scipy.optimize as optimization

# ## OPT - Waterfilling

# In[3]:


## Water-filling Algorithm for sorted demands
def waterfilling_sorted(d,b):
    n = np.size(d)
    allocations = np.zeros(n)
    bundle_remaining = b
    for i in range(n):
        equal_allocation = bundle_remaining/(n-i)
        if d[i]<equal_allocation:
            allocations[i] = bundle_remaining if i==n-1 else d[i]
        else:
            allocations[i] = equal_allocation
        bundle_remaining -= allocations[i]
    return allocations


# In[4]:


## Water-filling Algorithm for general demands
def waterfilling(d,b):
    n = np.size(d)
    sorted_indices = np.argsort(d)
    sorted_demands = np.sort(d)
    sorted_allocations = waterfilling_sorted(sorted_demands, b)
    allocations = np.zeros(n)
    for i in range(n):
        allocations[sorted_indices[i]] = sorted_allocations[i]
    return allocations


# In[5]:


## Tests
assert list(waterfilling(np.zeros(0), 5)) == []
assert list(waterfilling(np.array([1,2,3,4]), 10)) == [1,2,3,4]
assert list(waterfilling(np.array([3,4,1,2]), 10)) == [3,4,1,2]
assert list(waterfilling(np.array([1,2,3,4]), 8)) == [1,2,2.5,2.5]
assert list(waterfilling(np.array([3,1,4,2]), 8)) == [2.5,1,2.5,2]
assert list(waterfilling(np.array([3,6,5,6]), 8)) == [2,2,2,2]


# ## Online Algorithms

# In[6]:
## Online Water-filling taking minimum of realized demand and B/n
def waterfilling_proportional(demands_realized,b):
    n = np.size(demands_realized)
    allocations = np.zeros(n)
    eq = 1 if n==0 else b/n
    bundle_remaining = b
    for i in range(n):
        if i!=n-1:
            allocations[i] = min(eq, demands_realized[i])
        else:
            allocations[i] = bundle_remaining
        bundle_remaining -= allocations[i]
    return allocations

## Online Water-filling taking minimum of realized demand and B/n
def waterfilling_proportional_remaining(demands_realized,b):
    n = np.size(demands_realized)
    allocations = np.zeros(n)
    bundle_remaining = b
    for i in range(n):
        if i!=n-1:
            allocations[i] = min(bundle_remaining/(n-i), demands_realized[i])
        else:
            allocations[i] = bundle_remaining
        bundle_remaining -= allocations[i]
    return allocations


## Tests
assert list(waterfilling_proportional(np.zeros(0), 5)) == []
assert list(waterfilling_proportional(np.array([1,2,3,4]), 10)) == [1,2,2.5,4.5]
assert list(waterfilling_proportional(np.array([3,4,1,2]), 10)) == [2.5,2.5,1,4]
assert list(waterfilling_proportional(np.array([1,2,3,4]), 8)) == [1,2,2,3]
assert list(waterfilling_proportional(np.array([3,1,4,2]), 8)) == [2,1,2,3]
assert list(waterfilling_proportional(np.array([3,6,5,6]), 8)) == [2,2,2,2]

## Tests
assert list(waterfilling_proportional_remaining(np.zeros(0), 5)) == []
assert list(waterfilling_proportional_remaining(np.array([1,2,3,4]), 10)) == [1,2,3,4]
assert list(waterfilling_proportional_remaining(np.array([3,4,1,2]), 10)) == [2.5,2.5,1,4]
assert list(waterfilling_proportional_remaining(np.array([1,2,3,4]), 8)) == [1,2,2.5,2.5]
assert list(waterfilling_proportional_remaining(np.array([3,1,4,2]), 8)) == [2,1,2.5,2.5]
assert list(waterfilling_proportional_remaining(np.array([3,6,5,6]), 8)) == [2,2,2,2]
assert list(waterfilling_proportional_remaining(np.array([4,5,3,6]), 16)) == [4,4,3,5]
assert list(waterfilling_proportional_remaining(np.array([9,10,2,1]), 16)) == [4,4,2,6]



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


def insert_sorted(lst, element):
    n = np.size(lst)
    if n==0:
        return np.array([element]),0
    if element<lst[0]:
        return np.append(element,lst),0
    if element>lst[n-1]:
        return np.append(lst,element),n
    left = 0
    right = n-1
    while left<right-1:
        mid_ind = int((left+right)/2)
        if element<lst[mid_ind]:
            right = mid_ind
        elif element > lst[mid_ind] :
            left = mid_ind
        if element == lst[mid_ind] or (element>lst[mid_ind] and element<lst[mid_ind+1]):
            return np.append(np.append(lst[:mid_ind+1],element),lst[mid_ind+1:]), mid_ind+1
    return np.append(np.append(lst[:left],element),lst[left:]), left


# In[10]:


def delete_sorted(lst,element):
    n = np.size(lst)
    if element==lst[0]:
        return lst[1:]
    if element==lst[n-1]:
        return lst[:-1]
    left = 0
    right = n-1
    while left<right-1:
        mid_ind = int((left+right)/2)
        if element<lst[mid_ind]:
            right = mid_ind
        elif element > lst[mid_ind] :
            left = mid_ind
        else:
            return np.append(lst[:mid_ind],lst[mid_ind+1:])


# In[18]:


## O(n^2) version of online algorithm that needs waterfilling evaluated multiple times
def waterfilling_dynamic(demands_predicted, demands_realized, b):
    n = np.size(demands_predicted)
    sorted_demands = np.sort(demands_predicted)
    allocations = np.zeros(n)
    bundle_remaining = b
    for i in range(n):
        sorted_demands = delete_sorted(sorted_demands, demands_predicted[i])
        new_sorted_list,index = insert_sorted(sorted_demands,demands_realized[i])
        if i<n-1:
            allocations[i] = min((waterfilling_sorted(new_sorted_list, bundle_remaining))[index],demands_realized[i])
        else:
            allocations[i] = bundle_remaining
        bundle_remaining -= allocations[i]
    return allocations
    


# In[19]:


## Tests 
assert list(waterfilling_online_2(np.zeros(0), np.zeros(0), 5)) == []
assert list(np.around(waterfilling_online_2(np.array([1,2,3,4]), np.array([5,5,5,5]), 11),2)) == [3,2.67,2.67,2.67]
assert list(waterfilling_online_2(np.array([4,5,3,6]), np.array([2,1,8,6]), 15)) == [2,1,6,6]
assert list(waterfilling_online_2(np.array([4,5,3,6]), np.array([9,10,2,1]), 15)) == [4,4,2,5]

assert list(waterfilling_dynamic(np.zeros(0), np.zeros(0), 5)) == []
assert list(np.around(waterfilling_dynamic(np.array([1,2,3,4]), np.array([5,5,5,5]), 11),2)) == [3,2.67,2.67,2.67]
assert list(waterfilling_dynamic(np.array([4,5,3,6]), np.array([2,1,8,6]), 15)) == [2,1,6,6]
assert list(waterfilling_dynamic(np.array([4,5,3,6]), np.array([9,10,2,1]), 15)) == [4,4,2,5]
assert list(waterfilling_dynamic(np.array([4,5,3,6]), np.array([9,10,2,1]), 30)) == [9,10,2,9]


# In[ ]:


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


# In[14]:


## Tests 
assert list(waterfilling_online_3(np.zeros(0), np.zeros(0), 5)) == []
assert list(np.around(waterfilling_online_3(np.array([1,2,3,4]), np.array([5,5,5,5]), 11),2)) == [3,2.67,2.67,2.67]
assert list(waterfilling_online_3(np.array([4,5,3,6]), np.array([2,1,8,6]), 15)) == [2,1,6,6]
assert list(waterfilling_online_3(np.array([4,5,3,6]), np.array([9,10,2,1]), 15)) == [4,4,2,5]


# ## Objective Functions

# In[15]:


## Calculate log of Nash welfare for objective function
def objective_nash_log(demands, allocation):
    welfare_sum = 0
    for i in range(np.size(demands)):
        welfare_sum += np.log(min(1,allocation[i]/demands[i]))
    return welfare_sum


# In[ ]:


## Calculate log of Nash welfare for objective function
def objective_nash(demands, allocation):
    welfare_product = 1
    n=np.size(demands)
    for i in range(n):
        welfare_product = welfare_product*min(1,allocation[i]/demands[i])
    return welfare_product**1/n

## Calculate log of Nash welfare for objective function
def objective_nash_mod(demands, allocation):
    welfare_product = 1
    n=np.size(demands)
    for i in range(n):
        welfare_product = welfare_product*min(1,allocation[i]/demands[i])
    return welfare_product

## Calculate log of Nash welfare for objective function
def objective_sum(demands, allocation):
    welfare_sum = 0
    n=np.size(demands)
    for i in range(n):
        welfare_sum = welfare_sum+min(1,allocation[i]/demands[i])
    return welfare_sum

# ## Experiment

# In[16]:


def make_demands_uniform_distribution(num_towns, demand_ranges):
    demands = np.zeros(num_towns)
    expected_demands = np.zeros(num_towns)
    for i in range(num_towns):
        demands[i] = np.random.uniform(0, demand_ranges[i])
        expected_demands[i] = demand_ranges[i]/2
    return demands, expected_demands


def make_demands_gaussian_distribution(num_towns, demand_means):
    demands = np.zeros(num_towns)
    expected_demands = np.zeros(num_towns)
    for i in range(num_towns):
        demands[i] = np.random.normal(demand_means[i], demand_means[i]/5)
    return demands, demand_means


# In[17]:


def make_demands_exponential_distribution(num_towns, demand_means):
    demands = np.zeros(num_towns)
    expected_demands = np.zeros(num_towns)
    for i in range(num_towns):
        demands[i] = np.random.exponential(demand_means[i])
        expected_demands[i] = demand_means[i]
    return demands, expected_demands

