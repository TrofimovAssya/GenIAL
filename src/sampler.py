'''
Created on Oct 4, 2015

@author: bighouse
'''

from collections import Counter
from src.main import Processor, FileHandler
from sets import Set
import random



def generate_indices(total, sample, test):
        test_indices = Set()
        train_indices = Set()
        accu = 0
        while accu < sample + test :
            index = random.randint(0, total)
            if index in train_indices or index in test_indices :
                continue
            accu += 1
            if len(train_indices) < sample :
                train_indices.add(index)
            elif len(test_indices) < test :
                test_indices.add(index)
        return (train_indices, test_indices)

print generate_indices(1000, 100, 10)            

class Sampler(Processor):

    def __init__(self,
                 n,
                 sample = 100000,
                 test = 0.1,
                 training_file = "train_sample",
                 test_file = "test_sample"):
        self.test = test
        self.sample = generate_indices(n, sample, test * sample)
        self.training_file = open(training_file, "a")
        self.test_file = open(test_file, "a")
        self.line_number = 0
        
    def register(self, x):
        self.samfile = x
    
    def process(self, x):
        self.line_number += 1
        rname = self.samfile.getrname(x.tid)
        seq = x.seq
        if len(seq) < 40 :
            return
        if self.line_number in self.sample[0] :
            self.write_to_file(seq, rname, self.training_file)
        elif self.line_number in self.sample[1] :
            self.write_to_file(seq, rname, self.test_file)
        
    def write_to_file(self, data, clazz, f):
        f.write(data + "###" + clazz + "\n")
            
    def report(self):
        self.test_file.close()
        self.training_file.close()
        
def run():
    fh = FileHandler(Sampler(160000000, sample=10000))
    fh.from_file("/home/bighouse/data/ED06.bam")
        
run()
        
        
        