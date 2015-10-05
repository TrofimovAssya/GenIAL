#!/usr/bin/python
# -*- coding: utf-8 -*-

import pysam
import re
from StatsCollector import StatsCollector

class Vectorizer(object):

    def __init__(self, default_unknown = 0):
        self.default_unknown = default_unknown
    
    def encode_letter(self, letter):
        if letter == "T" :
            yield 1
            yield 0
            yield 0
            yield 0
        elif letter == "G" :
            yield 0
            yield 1
            yield 0
            yield 0
        elif letter == "C" :
            yield 0
            yield 0
            yield 1
            yield 0
        elif letter == "A":
            yield 0
            yield 0
            yield 0
            yield 1
        else :
            yield self.default_unknown
            yield self.default_unknown
            yield self.default_unknown
            yield self.default_unknown
            
    def encode(self, gene):
        ls = []
        for letter in gene :
            for vector_value in self.encode_letter(letter) :
                ls.append(vector_value)
        return ls

class Processor(object):
    def process(self, x) :
        pass
    
    def report(self):
        return None
    
    def register(self):
        return None

def is_nucleus_rname(s):
        return re.match("^chr[0-9]{1,2}$", s)

class StatsProcessor(Processor):
    
    def __init__(self, collector = StatsCollector()):
        self.collector = collector
        self.samfile = None
    
    def process(self, x) :
        if self.samfile is None :
            raise Exception("Cannot process without registering a samfile!")
        if is_nucleus_rname(self.samfile.getrname(x.tid)) :
            rname = self.samfile.getrname(x.tid)
            seq = x.seq
            if len(seq) < 5 :
                self.collector.add(len(seq), rname + "_short")
                self.collector.add(len(x.seq), rname)
                
    def register(self, x):
        self.samfile = x
        
    def report(self):
        print "label", "avg_length", "count", "min", "max"
        for label in self.collector.stats :
            print label, self.collector.stats[label][0], self.collector.stats[label][1],
            self.collector.stats[label][2], self.collector.stats[label][3]
        print "global min : ", self.collector.global_min
        print "global max : ", self.collector.global_max
        print "total : ", self.collector.total
        return None

class FileHandler(object):

    def __init__(self, processor = StatsProcessor()) :
        self.processor = processor

    def from_file(self, url):
        '''
        Tu peux définir un filtre sur les lignes que tu désires retenir.
        '''
        samfile = pysam.AlignmentFile(url, "rb")
        self.processor.register(samfile)
        for x in samfile.fetch():
            self.processor.process(x)
        return self.processor.report()
        
def example() :
    FileHandler().from_file("/home/bighouse/school/learning/ED06.bam")
    
