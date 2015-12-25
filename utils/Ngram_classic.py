import sys
import numpy as np
import os.path
import random
import cPickle as pickle
from collections import Counter


print sys.argv

def prep_Ngrams(filename, counterdict):
    ch = filename.split(".txt")[0]
    print ch
    with open(filename ,"r") as f:
        lines = [line.strip().split() for line in f]
        Ngrams[l[0]] = Counter({ch:int(line[1]) for line in lines})
        

    return Ngrams

#### be nice with Ugene
print "Hello! I'm UGene"
while(True):
    hi = raw_input("say hello!")
    if not len(hi)==0:
        print ":)"
        break
    else:
        print "c'mon, try again!"
        
print "counting probabilities...."

n = 15
Ngrams = {}

prep_Ngrams("x",Ngrams)

### this is the part where Ugene gets all chatty....    
print (" I can either analyse by combined max probability(CMP) or by best chromosome fit (BCF)")
t = raw_input("Which type of analysis you want?")
testfile = raw_input("What is the test file name?")
print "calculating...."


#### classifying test file reads according to user specs
with open(testfile) as tf:
    lines = [line.strip().split() for line in f]
correct = 0
total = 0
for l in lines:
    target = l[0].split("_")[0]
    read = l[1]
    if len(read)<n:
        continue
    kmer = [read[i:i+n] for i in range(0,len(read)-n)]
    total+=1
    
    if t == "CMP":
        probs = [max(Ngrams[k]) for k in kmer]
        c = Counter(probs)
        answer = max(c)            

    elif t == "BCF":
        probs = sum([Ngrams[k] for k in kmer])
        answer = max(probs)

    else:
        print "Sorry, don't know what that is. Goodbye"
        break
    print target==answer
    if target==answer:
        correct+=1


print "I predicted"+str(correct)+"out of "+str(total)+" total reads"
print "accuracy: "+str(correct/total*100)" %"

    
    
