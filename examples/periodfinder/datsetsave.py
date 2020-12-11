# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 20:50:40 2020

@author: sgtbh
"""


from datasetgen import datasetgen


l=str(datasetgen(1000,20000,10))


text_file = open("out/dataset_1000_20000_10.txt","w")
n=text_file.write(l)
text_file.close()
