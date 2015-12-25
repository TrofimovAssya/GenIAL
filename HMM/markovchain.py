import fileinput
import numpy as np
import pandas as pd
import os.path
import random
import itertools as it
import cPickle as pickle

def create_chain(row,col):
    chains = np.zeros((row,col))
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

    
def encode(s):
    s = [i for i in s]
    
    acc = 0
    for i in s:
        acc+=(code[i])
        acc*=4
    return acc
    
# generator makes all possible combinations for 5-nucleotide contigs
n = 5
temp = list(it.product('ACGT',repeat=n))
ix = ["".join(temp[i]) for i in range(len(temp))]
code = {"A":0,"C":1, "G":2, "T":3}

#create a dictionnary containing all markov chains (order 6) for each chromosome
Mchains = {}
Counts = {}
k = 1
for line in fileinput.input():
    print k
    l = line.strip("\n")
    l = l.split("\t")
    if not ( l[0] in Mchains.keys()):
        Mchains[l[0]] = create_chain(4*(4**n),4)
        Counts[l[0]] = create_chain(4*(4**n),1)
    l[1] = l[1].replace("N",randomN())
    for i in range(0,len(l[1])-5):
        dicodon = l[1][i:i+6]
        current = dicodon[:-1]
        nextc = dicodon[-1]
        Mchains[l[0]][encode(current),code[nextc]]+=1
        Counts[l[0]][encode(current),0] +=1
    k+=1
pickle.dump(Mchains,open("Mchains.p","w"))
pickle.dump(Counts,open("Mcounts.p","w"))
