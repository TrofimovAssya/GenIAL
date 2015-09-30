#!/usr/bin/python
# -*- coding: utf-8 -*-

import pysam
import re
from vectorize.StatsCollector import StatsCollector

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


def nucleus_chromosome(s):
    re.match("^chr[0-9]{1,2}$")

class Processor(object):

    def __init__(self, vectorizer = Vectorizer()) :
        self.vectorizer = vectorizer

    def is_nucleus_rname(self, s):
        return re.match("^chr[0-9]{1,2}$", s)

    def from_file(self, url, pred = lambda x : True):
        '''
        Tu peux définir un filtre sur les lignes que tu désires retenir.
        '''
        samfile = pysam.AlignmentFile(url, "rb")
        linecount = 0
        collector = StatsCollector()
        for x in samfile.fetch():
            linecount += 1
            if self.is_nucleus_rname(samfile.getrname(x.tid)) and pred(x) :
                rname = samfile.getrname(x.tid)
                seq = x.seq
                if len(seq) < 5 :
                    collector.add(len(seq), rname + "_short")
                collector.add(len(x.seq), rname)
        print "label", "avg_length", "count", "min", "max"
        for label in collector.stats :
            print label, collector.stats[label][0], collector.stats[label][1], collector.stats[label][2], collector.stats[label][3]
        print "global min : ", collector.global_min
        print "global max : ", collector.global_max
        print linecount.__repr__() + " lines."
    

Processor().from_file("/home/bighouse/school/learning/ED06.bam")
    
