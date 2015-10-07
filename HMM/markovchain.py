import fileinput
import numpy as np
import pandas as pd
import os.path
import random
import itertools as it
import cPickle as pickle

def create_chain(ix,cols):
    chains = pd.DataFrame(index = ix, columns = cols)
    chains = chains.fillna(0)
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

# generator makes all possible combinations for 5-nucleotide contigs
temp = list(it.product('ACGT',repeat=5))
ix = ["".join(temp[i]) for i in range(len(temp))]

#create a dictionnary containing all markov chains (order 6) for each chromosome
Mchains = {}
Counts = {}
k = 1
for line in fileinput.input():
    print k
    l = line.strip("\n")
    l = l.split("\t")
    if not ( l[0] in Mchains.keys()):
        Mchains[l[0]] = create_chain(ix,["A","C","G","T"])
        Counts[l[0]] = create_chain([ix],["count"])
    l[1] = l[1].replace("N",randomN())
    for i in range(0,len(l[1])-5):
        dicodon = l[1][i:i+6]
        current = dicodon[:-1]
        nextc = dicodon[-1]
        if nextc == "N":
            nextc = randomN()
        Mchains[l[0]][nextc].ix[current]+=1
        Counts[l[0]]['count'].ix[current] +=1
    k+=1
pickle.dump(Mchains,open("Mchains.p","w"))
pickle.dump(Counts,open("Mcounts.p","w"))
