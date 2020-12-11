# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 21:17:57 2020

@author: sgtbh
"""
from training import dnn


dataset=input("dataset file :")
sample_point=int(input("number of samplepoints:"))
num_layer=int(input("number of layers in the network : "))
batch_size=int(input("batch-size(preferably in the range(50,100): "))
epoch=int(input("epoch :"))
sparsity=int(input("sparsity :"))


dnn(dataset,sample_point,num_layer,epoch,batch_size,sparsity)