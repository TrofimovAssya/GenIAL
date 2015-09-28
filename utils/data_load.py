import sys
import numpy as np
import pandas as pd

def data_load(path):
    reads = path+"read"
    chrom = path+"chr"
    data = pd.read_csv(reads, sep='\t')
    answer = pd.read_csv(chrom,sep='\t')

    
    merged = data.join(answer)
    print merged
    tsize = 0.7
    tesize = 0.15
    vsize = 0.15
    b = len(merged.index)
    
    test = merged.sample(n = int(tesize*b))
    merged = merged[~merged.isin(test)].dropna()

    valid = merged.sample(n = int(vsize*b))
    merged = merged[~merged.isin(valid)].dropna()

    train = merged.sample(n=len(merged.index))

    return train,test,valid


