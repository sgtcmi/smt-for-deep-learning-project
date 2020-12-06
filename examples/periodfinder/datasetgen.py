#!/usr/bin/env python
# coding: utf-8

# In[229]:


import numpy as np
import math
from scipy import signal
import matplotlib

import matplotlib.pyplot as plt
import random


# In[230]:


def dot(v1, v2):
    return sum(x*y for x,y in zip(v1,v2))
def F1(T,t):                                               #sin
    return math.sin((2*np.pi*(1.0/T)*t))
def F2(T,t):                                               #sawtooth
    return(signal.sawtooth(2 * np.pi * (1.0/T) * t))
def F3(T,t):                                               #square
    return(signal.square(2 * np.pi * (1.0/T) * t))

def sum_to_x(n, x):
    values = [0.0, x] + list(np.random.uniform(low=0.0,high=x,size=n-1))
    values.sort()
    return [values[i+1] - values[i] for i in range(n)]

def ffactor(n):
    
    for i in range(2,n+1):
        if n%i==0:
            return(i) 
def primfact(n):
    x={}
    while n>1:
        
        m=ffactor(n)
        if m in x.keys():
            x[m]=x[m]+1 
        else:
            x[m]=1
        n=n//m 
    return(x)     
    

    
    
def revlcm(T):
    d=primfact(T)
    T1,T2,T3=(1,1,1)
    for i in d.keys():
        p=random.randint(1,d[i])
        q=random.randint(1,d[i])
        r=random.randint(1,d[i])
        if max(p,q,r)!=d[i]:
            r=d[i]
        T1=T1*(i**p)
        T2=T2*(i**q)
        T3=T3*(i**r)
        
            
    return(T1,T2,T3)      

    
    
def datapointgen(T,t):
    T1,T2,T3=revlcm(T)
    samples=[]
    l=sum_to_x(3,1)
    
    for i in range(t):
        samples=samples+[dot(l,[F1(T1,i),F2(T2,i),F3(T3,i)])]
        
    return(samples)
    
def plot(T,t):
    plt.plot(list(range(t)),datapointgen(T,t))    


# In[224]:


plot(128*3,1000)


# In[231]:


def datasetgen(t,n):
    dataset=[]
    label=[]
    for i in range(n):
        T=random.randint(20,499)
        dataset.append(datapointgen(T,t))
        label.append(T)
        
    return(dataset,label)
    
    


# In[232]:





# In[ ]:




