
from datasetgen import datasetgen

sp=int(input("number of sample points to be generated :"))
dp=int(input("number of datapoints needed : "))
s=int(input("sparsity : "))
dataset_filename=input("desired filename :")
l=str(datasetgen(sp,dp,s))



text_file = open(dataset_filename,"w")
n=text_file.write(l)
text_file.close()