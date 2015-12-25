'''
@author: bighouse
'''

from collections import Counter
import cProfile, pstats, StringIO
import time
current_milli_time = lambda: int(round(time.time() * 1000))

class BayesModelRunner():
    
    def __init__(self, classifier, train_file, test_file, validation_file, split_size = 0, profiling = False, verbose = True):
        self.classifier = classifier
        self.test_file = test_file
        self.validation_file = validation_file
        self.train_file = train_file
        self.split_size = split_size
        self.profiling = profiling
        self.verbose = verbose
        if profiling :
            self.pr = cProfile.Profile()

    def validate(self):
        with open(self.train_file, "r") as f:
            for l in f.readlines() :
                splitted = l.split("###")
                chromo = splitted[0]
                length = len(chromo)
                if self.split_size > 0 and length > self.split_size :
                    self.classifier.train(chromo[:length/2], splitted[1][:-1])
                    self.classifier.train(chromo[length/2:], splitted[1][:-1])
                else :
                    if self.profiling :
                        self.pr.enable()
                    self.classifier.train(chromo, splitted[1][:-1])
                    if self.profiling :
                        self.pr.disable()
                        s = StringIO.StringIO()
                        sortby = 'cumulative'
                        ps = pstats.Stats(self.pr, stream=s).sort_stats(sortby)
                        ps.print_stats()
                        print s.getvalue()
        
        self.classifier.prepare()
        return self.classify(self.validation_file)
        
    
    def classify(self, file_to_classify):
        
        if self.verbose :
            print 'Classifying'
        
        confusion_matrix = Counter()
        num_lines = 0
        with open(file_to_classify, "r") as f :
            for l in f.readlines() :
                if self.verbose :
                    time1 = current_milli_time()
                num_lines += 1
                splitted = l.split("###")
                predicted = self.classifier.classify(splitted[0])
                actual = splitted[1][:-1]
                confusion_matrix[(predicted, actual)] += 1
                if self.verbose and num_lines % 10000 == 0 :
                    print 'classified ', num_lines, ' lines.'
                    print current_milli_time() - time1, 'ms'
                
        return confusion_matrix
    
    def test(self):
        return self.validate(self.test_file)