import sys
import numpy as np
import os.path
import random
import cPickle as pickle
import jellyfish




test_file = sys.argv[1]
''' analysis types possible for the predictor:
CMP - Combined Max Probability (all maximums)
BCF - Best chromosome fit '''

totals = pickle.load(open("kmersum.p","r"))
jfDB = pickle.load(open("jfDB.p","r"))
apriori = pickle.load(open("class_apriori.p","r"))

#fonction qui fait la query dans le fichier DB de jf
def query_chr_db(filename,mer):
    qf = jellyfish.QueryMerFile(filename)
    m = jellyfish.MerDNA(mer)
    return qf[m]

'''returns the observed log2 probability of the kmer within the chromome'''
def get_prob(mer,chrom):
    filename = jfDB[chrom]
    count = query_chr_db(filename,i)
    if count==0:
        return 0
    total = totals[chrom]
    prior = apriori[chrom]
    prob = np.log2((float(count)/float(total))*float(prior))
    return prob

'''returns the observed log2 probability of the kmer within the chromome'''
def get_prob_set(mer_tab,chrom):
    filename = jfDB[chrom]
    psum = 0
    for i in mer_tab:
        count = query_chr_db(filename,mer)
        if count==0:
            continue
        psum+=count
    total = totals[chrom]
    prior = apriori[chrom]
    prob = np.log2((float(psum)/float(total))*float(prior))
    return prob



'''CMP - Combined Max Probability:
each kmer within the read is queried for all chromosomes
the highest probability is retained.
the maximum contributing chromosome is the predicted class
'''
def predict_CMP(read,n):
    maxchrom = []
    i = 0
    if len(read)<n:
        return None
    while not i==len(read)-n:
        mer = read[i:i+n]
        merprob = {}
        for k in jfDB.keys():
            p = get_prob(mer,k)
            if not p==0:
                merprob[k] = p
        if not len(merprob)==0:
            maxchrom.append(max(merprob,key = merprob.get))
        i+=1
    if not len(maxchrom)==0:
        return max(maxchrom,key = maxchrom.count)
    else:
        return None
    
def predict_BCF(read,n):
    chromprob = {}
    i = 0
    if len(read)<n:
        return None
    mer_tab = []
    #generating the kmer array for the read
    while not i==len(read)-n:
        mer_tab.append(read[i:i+n])
        i+=1
    #evaluating the probabilities for each kmer in the chromosome
    for k in jfDB.keys():
        p = get_prob_set(mer_tab,k)
        if not p==0:
            chromprob[k] = p

    if not len(chromprob)==0:
        return max(chromprob,key = chromprob.count)
    else:
        return None
    
    


''' error rates:
train error: error rate on reads included in the training
test error: error rate on reads never seen in training
'''


########## tests here
g = open("results.txt","w")
#def test_model(filename):
f = open(test_file,"r")
count = 0
correct = 0
for line in f:
    l = line.strip().split("\t")
    read = l[1]
    ch = l[0]
    y = predict_CMP(read,15)
    if y==ch:
        correct+=1
    if not y==None:
        count+=1
    if y == None:
        print("don't know")
        g.write("don't know")
        g.write("\n")
    else:
        g.write(str(ch)+" "+str(y))
        g.write("\n")
        g.write(str(correct)+" / "+str(count))
        g.write("\n")
        
        print(ch,y)
        print(str(correct)+" / "+str(count))
        
    print("*********fin***********")
    print("Accuracy:")
    g.write("*********fin***********")
    g.write("Accuracy:")

    r = float(correct)/float(count)*100
    g.write(str(r))
    g.write("\n")
    print r

#    return r

#result = test_model(test_file)
g.close()

f.close()
