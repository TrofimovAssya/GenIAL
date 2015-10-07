import fileinput
import numpy as np
import os.path
import random

random.seed(1234)


sampl = open("sample_bam06distrib.txt","w")



for line in fileinput.input():
    l = line.split("\t")
    ch = l[2]
    re = l[9]
    if random.random()<0.0001:
        sampl.write("\t".join([ch,re]))
        sampl.write("\n")
