'''
Created on Dec 15, 2015

@author: bighouse
'''

from BayesModelRunner import BayesModelRunner
from ngram import NgramSequenceClassifier
from interpolation import InterpolationClassifier
from laplace import AddDeltaClassifier

def bayes_sample_run(train_file, test_file, validation_file):
    max_k = 0
    max_success_rate = 0.0
    max_result = None
    for k in range(7, 12) :
        runner = BayesModelRunner(NgramSequenceClassifier(k),
                                  train_file,
                                  test_file,
                                  validation_file,
                                  verbose=False)
        result = runner.validate()
        
        success = 0
        total = 0
        for index, count in result.iteritems() :
            total += count
            if index[0] == index[1] :
                success += count
        
        if total == 0 :
            print "Something went wrong for k = ", k
            continue        
        
        success_rate = (success + 0.0) / total
        if success_rate > max_success_rate :
            max_k = k
            max_success_rate = success_rate
            max_result = result
        
        print "For k = ", k
        print "Success rate ", success_rate
        print result
    print "Best run is ", max_k, max_success_rate, max_result


def frange(start, end=None, inc=None):
    "A range function, that does accept float increments..."
    if end == None:
        end = start + 0.0
        start = 0.0
    if inc == None:
        inc = 1.0
    L = []
    while 1:
        next = start + len(L) * inc
        if inc > 0 and next >= end:
            break
        elif inc < 0 and next <= end:
            break
        L.append(next)
    return L

def laplace_sample_run(train_file, test_file, validation_file):
    max_k = 0
    max_success_rate = 0.0
    max_result = None
    for k in range(10, 12) :
        for delta in frange(0.2, 1.1, 0.2) :
            print "For k = ", k, 'and', delta
            runner = BayesModelRunner(AddDeltaClassifier(k, 4, delta=delta),
                                  train_file,
                                  test_file,
                                  validation_file,
                                  verbose=False,
                                  )
            result = runner.validate()
        
            success = 0
            total = 0
            for index, count in result.iteritems() :
                total += count
                if index[0] == index[1] :
                    success += count
        
            if total == 0 :
                print "Something went wrong for k = ", k
                continue        
        
            success_rate = (success + 0.0) / total
            if success_rate > max_success_rate :
                max_k = k
                max_success_rate = success_rate
                max_result = result
        
            print "For k = ", k
            print "Success rate ", success_rate
            print result
    print "Best run is ", max_k, max_success_rate, max_result, delta

def interpolation_sample_run(train_file, test_file, validation_file):
    max_k = 0
    max_success_rate = 0.0
    max_result = None
    for k in range(10, 12) :
        for delta in frange(0.65, end=1, inc=0.1) :
            weights = [delta, 1 - delta]
            print "For k = ", k, 'and', weights
            runner = BayesModelRunner(InterpolationClassifier(k, weights),
                                  train_file,
                                  test_file,
                                  validation_file,
                                  verbose=False)
            result = runner.validate()
        
            success = 0
            total = 0
            for index, count in result.iteritems() :
                total += count
                if index[0] == index[1] :
                    success += count
        
            if total == 0 :
                print "Something went wrong for k = ", k
                continue        
        
            success_rate = (success + 0.0) / total
            if success_rate > max_success_rate :
                max_k = k
                max_success_rate = success_rate
                max_result = result
        
            print "Success rate ", success_rate
            #print result
        
    print "Best run is ", max_k, max_success_rate, max_result    

# 
for suffices in [['06', '06']] :
    print '+ + + + + FOR RUN train/validate', suffices
    interpolation_sample_run("../data/train_sample_" + suffices[0],
                            "../data/test_sample_" + suffices[1],
                            "../data/validation_sample_" + suffices[1])
    laplace_sample_run("../data/train_sample_" + suffices[0],
                            "../data/test_sample_" + suffices[1],
                            "../data/validation_sample_" + suffices[1])

