import fileinput
import numpy as np
import os.path

def onehot_convert(read):
    for i in read:
        if i=="A":
            yield 0
            yield 0
            yield 0
            yield 1
        elif i=="C":
            yield 0
            yield 0
            yield 1
            yield 0
        elif i=="G":
            yield 0
            yield 1
            yield 0
            yield 0
        elif i=="T":
            yield 1
            yield 0
            yield 0
            yield 0
        else:
            yield 0
            yield 0
            yield 0
            yield 0

def strsplitter(string,l,table):
    if len(string)<l:
        return table
    elif len(string)>l:
        table.append(string[0:l+1])
        return strsplitter(string[l+1:len(string)],l,table)
    elif len(string)==l:
        table.append(string)
        return table
    

for line in fileinput.input():
    l = line.split("\t")
    chrom = l[0]
    temp = strsplitter(l[1].split("\n")[0],50,[])
    if temp:
        for j in xrange(len(temp)):
            if(len(temp[j])>0):
                x = []
                for k in onehot_convert(temp[j]):
                    x.append(k)
                print chrom,";",x



