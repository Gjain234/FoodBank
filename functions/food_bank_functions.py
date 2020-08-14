#  Food Bank Problem

import sys
import numpy as np
import plotly.express as px
import pandas as pd
import scipy.optimize as optimization

# ## OPT - Waterfilling


## Water-filling Algorithm for sorted demands
def waterfilling_sorted(d,b):
    n = np.size(d)
    allocations = np.zeros(n)
    bundle_remaining = b
    for i in range(n):
        equal_allocation = bundle_remaining/(n-i)
        if d[i]<equal_allocation:
            allocations[i] = min(d[i], bundle_remaining) if i==n-1 else d[i]
        else:
            allocations[i] = equal_allocation
        bundle_remaining -= allocations[i]
    return allocations

## Water-filling Algorithm for sorted demands
def waterfilling_sorted_waste(d,b):
    n = np.size(d)
    allocations = np.zeros(n)
    bundle_remaining = b
    for i in range(n):
        equal_allocation = bundle_remaining/(n-i)
        if d[i]<equal_allocation:
            allocations[i] = d[i]
        else:
            allocations[i] = equal_allocation
        bundle_remaining -= allocations[i]
    return allocations

## Water-filling Algorithm for sorted demands and weights for bucket width
def waterfilling_sorted_weights(demands, weights, budget):
    n = np.size(demands)
    allocations = np.zeros(n)
    budget_remaining = budget
    width = np.sum(weights)

    for i in range(n):

        equal_allocation = budget_remaining / width
        if demands[i]<equal_allocation:
            allocations[i] = min(budget_remaining, demands[i]) if i == n-1 else demands[i]
        else:
            allocations[i] = equal_allocation

        budget_remaining -= allocations[i]*weights[i]
        width -= weights[i]

    return allocations



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

## Water-filling Algorithm for general demands
def waterfilling_waste(d,b):
    n = np.size(d)
    sorted_indices = np.argsort(d)
    sorted_demands = np.sort(d)
    sorted_allocations = waterfilling_sorted_waste(sorted_demands, b)
    allocations = np.zeros(n)
    for i in range(n):
        allocations[sorted_indices[i]] = sorted_allocations[i]
    return allocations


## Tests
assert list(waterfilling(np.zeros(0), 5)) == []
assert list(waterfilling(np.array([1,2,3,4]), 10)) == [1,2,3,4]
assert list(waterfilling(np.array([3,4,1,2]), 10)) == [3,4,1,2]
assert list(waterfilling(np.array([1,2,3,4]), 8)) == [1,2,2.5,2.5]
assert list(waterfilling(np.array([3,1,4,2]), 8)) == [2.5,1,2.5,2]
assert list(waterfilling(np.array([3,6,5,6]), 8)) == [2,2,2,2]


# ## Online Algorithms

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
            allocations[i] = min(demands_realized[i], bundle_remaining)
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
            allocations[i] = min(bundle_remaining, demands_realized[i])
        bundle_remaining -= allocations[i]
    return allocations






def insert_sorted(lst, element):
    n = np.size(lst)
    if n==0:
        return np.array([element]),0
    if element<=lst[0]:
        return np.append(element,lst),0
    if element>=lst[n-1]:
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

## O(n^2) version of online algorithm that needs waterfilling evaluated multiple times
def waterfilling_dynamic_waste(demands_predicted, demands_realized, b):
    n = np.size(demands_predicted)
    sorted_demands = np.sort(demands_predicted)
    allocations = np.zeros(n)
    bundle_remaining = b
    for i in range(n):
        sorted_demands = delete_sorted(sorted_demands, demands_predicted[i])
        new_sorted_list,index = insert_sorted(sorted_demands,demands_realized[i])
        allocations[i] = min((waterfilling_sorted(new_sorted_list, bundle_remaining))[index],demands_realized[i], bundle_remaining)
        bundle_remaining -= allocations[i]
    return allocations

# In[19]:


assert list(waterfilling_dynamic(np.zeros(0), np.zeros(0), 5)) == []
assert list(np.around(waterfilling_dynamic(np.array([1,2,3,4]), np.array([5,5,5,5]), 11),2)) == [3,2.67,2.67,2.67]
assert list(waterfilling_dynamic(np.array([4,5,3,6]), np.array([2,1,8,6]), 15)) == [2,1,6,6]
assert list(waterfilling_dynamic(np.array([4,5,3,6]), np.array([9,10,2,1]), 15)) == [4,4,2,5]
assert list(waterfilling_dynamic(np.array([4,5,3,6]), np.array([9,10,2,1]), 30)) == [9,10,2,9]


## Waterfilling using weighted bars
def waterfilling_weights(weights, sorted_distribution, demands_realized, budget):
    n = np.size(demands_realized)
    distribution_size = np.size(sorted_distribution)
    distribution_weighted = weights*sorted_distribution
    allocations = np.zeros(n)
    budget_remaining = budget
    for i in range(n):
        new_sorted_list,index = insert_sorted(sorted_distribution,demands_realized[i])
        if i<n-1 :
            allocations[i] = min((waterfilling_sorted_weights(new_sorted_list, weights,budget_remaining,index,n-i-1))[index],demands_realized[i])
        else:
            allocations[i] = budget_remaining
        budget_remaining -= allocations[i]
    return allocations

## Waterfilling using weighted bars
def waterfilling_weights_waste(weights, supports, demands_realized, budget):
    n = np.size(demands_realized)
    allocations = np.zeros(n)
    budget_remaining = budget

    for i in range(n):
        # need to collect distribution on weights
        # and add in one for observed demand
        if i<n-1:

            support = supports[(i+1):n]
            vals = weights[(i+1):n]

            support = support.flatten()
            support = np.around(support, decimals=2)
            vals = vals.flatten()

            new_support, inverse_index = np.unique(support, return_inverse=True)

            new_vals = np.zeros(len(new_support))

            for j in range(len(vals)):
                new_vals[inverse_index[j]] += vals[j]

            if np.around(demands_realized[i]) in new_support:
                index = np.argmin(np.abs(new_support - demands_realized[i]))
                new_vals[index] += 1
            else:
                new_support, index = insert_sorted(new_support, demands_realized[i])
                new_vals = np.append(np.append(new_vals[:index],1),new_vals[index:])

            waterfilling_alloc = waterfilling_sorted_weights(new_support, new_vals, budget_remaining)
            # print('waterfilling_allocation: ' + str(waterfilling_alloc))
            allocations[i] = min(waterfilling_alloc[index],demands_realized[i])
        else:
            allocations[i] = min(budget_remaining,demands_realized[i])
        budget_remaining -= allocations[i]
    return allocations






def greedy(demands_realized,budget):
    n = np.size(demands_realized)
    allocations = np.zeros(n)
    budget_remaining = budget
    index = 0

    for i in range(n):
        if demands_realized[i] <= budget_remaining:
            allocations[i] = demands_realized[i]
            budget_remaining -= demands_realized[i]
        else:
            allocations[i] = budget_remaining
            budget_remaining = 0
    return allocations


def constant_threshold(demands_realized,budget,threshold):
    n = np.size(demands_realized)
    allocations = np.zeros(n)
    budget_remaining = budget
    for i in range(n):
        if demands_realized[i]<threshold:
            allocations[i] = min(budget_remaining,demands_realized[i])
        else:
            allocations[i] = min(budget_remaining,threshold)
        budget_remaining -= allocations[i]
    return allocations


# vector returning the maximum envy every town feels
def envy_vector(allocation, demands):
    n = np.size(demands)
    allocation_max = np.amax(allocation)
    envy = np.zeros(n)
    for i in range(n):
        envy[i] = min(allocation_max/demands[i], 1) - min(allocation[i]/demands[i],1)
    return envy


def proportionality(allocation,demands, budget):
    n = np.size(demands)
    prop = budget/n
    max_prop = np.zeros(n)
    for i in range(n):
        if allocation[i]<demands[i]:
            max_prop[i] = max(0,min(prop/demands[i],1) - min(allocation[i]/demands[i],1))
    return max_prop


def excess(allocation, budget):
    return (1/len(allocation))*(budget-np.sum(allocation))


def envy_utility(allocation, demands):
    n = np.size(demands)
    allocation_max = np.amax(allocation)
    envy = np.zeros(n)
    for i in range(n):
        envy[i] = min(1,allocation_max/demands[i]) - min(1,allocation[i]/demands[i])
    return envy

def proportionality_utility(allocation,demands, budget):
    n = np.size(demands)
    prop = budget/n
    max_prop = np.zeros(n)
    for i in range(n):
        max_prop[i] = min(1,prop/demands[i]) - min(1,allocation[i]/demands[i])
    return max_prop

# ## Objective Functions

demands1 = np.array([1,2,1,2,1])
demands2 = np.array([1,1,1,1,1])
demands3 = np.array([2,2,2,2,2])
demands4 = np.array([2,2,1,1,1])
demands4 = np.array([1,1,2,2,2])
demands4 = np.array([2,1,2,1,2])
budget = 7.5

assert list(envy_vector(np.array([1,1,1,1,1]),demands1)) == [0,0,0,0,0]
assert list(envy_vector(demands1,demands1)) == [0,0,0,0,0]
assert list(envy_vector(np.array([1,1,1,2,1]),demands1)) == [0,0.5,0,0,0]
assert list(envy_vector(np.array([1,1,1,2,0]),demands1)) == [0,0.5,0,0,1]
assert list(envy_vector(np.array([0,0,0,0,5]),demands1)) == [1,1,1,1,0]
assert list(envy_vector(np.array([2,2,2,2,2]),demands1)) == [0,0,0,0,0]
#############
assert list(proportionality(np.array([1,2,1,2,1]),demands1,budget)) == [0,0,0,0,0]
assert list(proportionality(np.array([1,1,1,1,1]),demands1,budget)) == [0,0.25,0,0.25,0]
assert list(proportionality(np.array([1.5,1.5,1.5,1.5,1.5]),demands1,budget)) == [0,0,0,0,0]
##############
assert excess(np.array([1,1,1,1,1]),10) == 5/5
assert excess(np.array([1,1,1,1,1]),1) == (-4)/5
assert excess(np.array([1,1,1,1,1]),5) == 0



## Calculate log of Nash welfare for objective function
def objective_nash_log(demands, allocation):
    welfare_sum = 0
    for i in range(np.size(demands)):
        welfare_sum += np.log(min(1,allocation[i]/demands[i]))
    return welfare_sum

def objective_nash_log_normalized(demands, allocation):
    welfare_sum = 0
    n = np.size(demands)
    for i in range(n):
        welfare_sum += np.log(min(1,allocation[i]/demands[i]))
    return welfare_sum/n

def objective_nash_log_vector(demands, allocation):
    n = np.size(demands)
    welfare_vector = np.zeros(n)
    for i in range(n):
        welfare_vector[i] = np.log(min(1,allocation[i]/demands[i]))
    return welfare_vector


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
