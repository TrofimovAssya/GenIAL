import fileinput
import numpy as np
import pandas as pd
import os.path
import random
import itertools as it


def create_chain(ix,cols):
    chains = pd.DataFrame(index = ix, columns = cols)
    chains = chains.fillna(0)
    return chains

# generator makes all possible combinations for 5-nucleotide contigs
temp = list(it.combinations_with_replacement('ACGT',5))
ix = ["".join(temp[i]) for i in range(len(temp))]

#create a dictionnary containing all markov chains (order 6) for each chromosome
Mchains = {}
Counts = {}

for line in fileinput.input():
    print line
    l = line.split("\t")
    if not ( l[0] in Mchains.keys()):
        Mchains[l[0]] = create_chain(ix,["A","C","G","T"])
        Mchain[l[0]]
        Counts[l[0]] = create_chain('count',["A","C","G","T"])
        
    #### examine all 
    for i in range(0,len(l[1]-5)):
        dicodon = l[1][i:i+6]
        current = dicodon[:-1]
        next = dicodon[-1]
        Mchains[l[0]][next].ix[current]+=1
        Counts[l[0]][next].ix['count'] +=1
        
pickle.dump(open("Mchains.p","w"))
pickle.dump(open("Mcounts.p","w"))
