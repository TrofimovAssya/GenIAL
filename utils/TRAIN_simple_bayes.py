import fileinput
import numpy as np
from scipy.spatial.distance import euclidean as euclid
import scipy
import pandas as pd
import os.path
import random
import itertools as it
import cPickle as pickle

from simple_prepchains import create_chain
from simple_prepchains import randomN
from simple_prepchains import create_index
from simple_prepchains import process_read
from simple_prepchains import poisson_prob


print "training...."

n=5
ix = create_index(n)

'''create a dictionnary containing all markov chains (order 6) for each chromosome'''
Mchains = {}
probchr = {}
a = 0
for line in fileinput.input():
    l = line.strip("\n")
    l = l.split("\t")
    l[0] = l[0].split("_")[0]
    if len(l[1])<=n:
        continue
    if not ( l[0] in Mchains.keys()):
        Mchains[l[0]] = create_chain(len(ix),1)
        probchr[l[0]] = 0
    temp = create_chain(len(ix),1)
    l[1] = l[1].replace("N",randomN())
    Mchains[l[0]]+=process_read(l[1],temp,n,ix)
    probchr[l[0]]+=1
sum_prob = float(np.sum(probchr.values()))

#calculating Poisson distribution parameters (mu) for each Markov chain
for k in Mchains.keys():
    Mchains[k] = Mchains[k]/float(sum(Mchains[k]))
    probchr[k] = probchr[k]/sum_prob
    


''' For each chromosome, the occurences are converted to 
probability tables. 
For each chromosome, a |1-distance|**2 is calculated as a metric'''
fname = "".join([str(n),"_Mchains.p"])
pickle.dump(Mchains,open((fname),"w"))

print "fitting model to training set..."
a = 0
success = np.zeros((len(Mchains.keys()),1))

for line in fileinput.input():
    l = line.strip("\n")
    l = l.split("\t")
    l[0] = l[0].split("_")[0]
    if len(l[1])<=n:
        continue
    target = l[0]
    
    answer = np.array(Mchains.keys())
    
    estimates = np.zeros( (len(Mchains.keys()),1))
    
    temp = create_chain(len(ix),1)
    l[1] = l[1].replace("N",randomN())
    temp = process_read(l[1],temp,n,ix)
    temp = temp/float(sum(temp))
    for k,val in enumerate(Mchains.keys()):
        prob = euclid(temp,Mchains[val])
        estimates[k] = np.log2(prob)
    best = np.argmin(estimates)
    print answer[best], target
    if answer[best] == target:
        a+=1
    
print "total of ",a, "fitted properly"
