# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 21:17:57 2020

@author: sgtbh

Command line usage:
    arg1    -   Model file name. If provided and a model exists at the given location, it
                loads the model, in which case all other arguments except the second one are
                ignored. If no model exists in the given location, it saves the model.
    arg2    -   Dataset file name
    arg3    -   Number of samples per point in the dataset
    arg4    -   Number of layers
    arg5    -   Batch size, set between 50-100 for best results
    arg6    -   Number of epochs
    arg7    -   Sparsity of the data
"""
from training import dnn
import sys
import os
from tensorflow.keras import models, layers

model_file = sys.argv[1]

print("Creating and training model")
dataset=sys.argv[2]
sample_point=int(sys.argv[3])
num_layer=int(sys.argv[4])
batch_size=int(sys.argv[5])
epoch=int(sys.argv[6])
sparsity=int(sys.argv[7])


mdl = dnn(dataset,sample_point,num_layer,epoch,batch_size,sparsity)
mdl.save(model_file)



