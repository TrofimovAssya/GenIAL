'''
Created on Oct 4, 2015
@author: bighouse
'''

from src.main import Processor, FileHandler
from sets import Set
import random

def generate_indices(total, sample, test, validation):
        test_indices = Set()
        train_indices = Set()
        validation_indices = Set()
        accu = 0
        while accu < sample + test + validation :
            index = random.randint(0, total)
            if index in train_indices or index in test_indices or index in validation_indices:
                continue
            accu += 1
            if len(train_indices) < sample :
                train_indices.add(index)
            elif len(test_indices) < test :
                test_indices.add(index)
            elif len(validation_indices) < validation :
                validation_indices.add(index)
        return (train_indices, test_indices, validation_indices)

class Sampler(Processor):

    def __init__(self,
                 n,
                 training_file,
                 test_file,
                 validation_file,
                 sample = 100000,
                 test = 0.1,
                 validation = 0.1
      ):
        self.test = test
        self.sample = generate_indices(n, sample, test * sample, validation * sample)
        self.training_file = open(training_file, "a")
        self.test_file = open(test_file, "a")
        self.validation_file = open(validation_file, "a")
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
        elif self.line_number in self.sample[2] :
            self.write_to_file(seq, rname, self.validation_file)
        
    def write_to_file(self, data, clazz, f):
        f.write(data + "###" + clazz + "\n")
            
    def report(self):
        self.test_file.close()
        self.training_file.close()
        self.validation_file.close()

ED06_path = "/home/bighouse/data/ED06.bam"
ED06_wc = 111344139

ED39_path = "/home/bighouse/data/ED39.bam"
ED39_wc = 0

def small_sample(wc, bamfile) :
    global ED06_wc
    fh = FileHandler(Sampler(ED06_wc, "../data/small_train_sample",
                             "../data/small_test_sample",
                             "../data/small_validation_sample",
                              sample=35000))
    fh.from_file(bamfile)
    
def normal_sample(wc, bamfile) :
    global ED06_wc
    fh = FileHandler(Sampler(ED06_wc,
                             "../data/train_sample",
                             "../data/test_sample",
                             "../data/validation_sample",
                              sample=350000))
    fh.from_file(bamfile)
        
normal_sample(ED06_wc, ED06_path)