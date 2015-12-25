import fileinput
import numpy as np
import pandas as pd
import scipy as sp
from scipy.misc import factorial
import os.path
import random
import itertools as it
import cPickle as pickle
import time

def create_chain(row,col):
    '''introducing pseudocounts here.
    When the probs are zero, we assume the
    probability is all equal (not null)'''

    chains = np.ones((row,col))
    return chains

def randomN ():
    a = random.random()
    if a<0.25:
        return "A"
    elif a<0.5:
        return "C"
    elif a<0.75:
        return "G"
    else:
        return "T"

    
'''generator makes all possible combinations for n-nucleotide contigs'''

def create_index(n):
    temp = list(it.product('ACGT',repeat=n+1))
    ix = ["".join(temp[i]) for i in range(len(temp))]
    ix = dict( (i,j) for (i,j) in zip(ix,xrange(len(ix))))    
    return ix

def process_read(read,temp,n,ix):
    for i in range( 0, (len(read)-n) ):
        current = read[i:i+n+1]
        temp[ix[current],0]+=1
        temp = temp/float(np.sum(temp))
        return temp
    
def poisson_prob(value,mu):
    nng = np.sum(value)
    lm = mu * nng
    prob = ((np.exp(-lm)) * (lm**(value))) / (factorial(value,exact = False))
    return prob
    

