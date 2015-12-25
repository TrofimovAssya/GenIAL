import sys
import numpy as np
import scipy
import pandas as pd
import os.path
import random
import itertools as it
import cPickle as pickle
from collections import Counter

from simple_prepchains import create_chain
from simple_prepchains import randomN
from simple_prepchains import create_index
from simple_prepchains import process_read
from simple_prepchains import poisson_prob

print "hello!"
print "counting probabilities...."

n = 15

Ngrams = {}

for i in sys.argv[1:]:
    ch = sys.argv[i].split(".txt")[0]
    Ngrams[ch] = Counter()
    
    f = open(sys.argv[i],"r")
    for l in f.readlines():
        l = l.split("\t")
        Ngrams[ch] = Ngrams[ch]+(l[0] = l[1])


    
