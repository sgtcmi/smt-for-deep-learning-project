
import numpy as np
import math
from scipy import signal
import random




#dot product of two vectors
def dot(v1, v2):
    return sum(x*y for x,y in zip(v1,v2))
# sin-wave function
def F1(T,t):                                               
    return math.sin((2*np.pi*(1.0/T)*t))
# sawtooth wave function
def F2(T,t):                                               
    return(signal.sawtooth(2 * np.pi * (1.0/T) * t))
# square wave function
def F3(T,t):                                               
    return(signal.square(2 * np.pi * (1.0/T) * t))

# Given a number x find n numbers such that their sum is x
def sum_to_x(n, x):
    values = [0.0, x] + list(np.random.uniform(low=0.0,high=x,size=n-1))
    values.sort()
    return [values[i+1] - values[i] for i in range(n)]
# prime factorization
def ffactor(n):
    
    for i in range(2,n+1):
        if n%i==0:
            return(i) 
# return a dictionary with the prime factors as key and their powers as values
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
    
#Given a number T(the time period) get three numbers T1 ,T2 ,T3 whose lcm is T   
    
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

   
# data point with t samples    
def datapointgen(T,t):
    T1,T2,T3=revlcm(T)
    samples=[]
    l=sum_to_x(3,1)
    
    for i in range(t):
        samples=samples+[dot(l,[F1(T1,i),F2(T2,i),F3(T3,i)])]
        
    return(samples)
# To plot the curve (superimposed waveform)    
def plot(T,t):
    plt.plot(list(range(t)),datapointgen(T,t))    





def datasetgen(t,n,sparsity):
    dataset=[]
    label=[]
    for i in range(n):
        print('Generating point %d'%i, end='\r')
        T=random.randrange(t//10,t//2,sparsity )
        
        dataset.append(datapointgen(T,t))
        label.append(T)
        
    return(dataset,label)
    
    





