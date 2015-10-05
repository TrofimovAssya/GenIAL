'''
Created on Oct 4, 2015

@author: bighouse
'''

from sets import Set
from collections import Counter
from src.main import FileHandler, Processor, is_nucleus_rname

# Proof of concept
# This algorithm is really naive, but may work an alphabet of size 4.
class BayesModel(object) :
    
    def __init__(self, k = 3):
        self.k = k
        self.counters = {}
        self.total = 0
        
    def train(self, data):
        l = len(data)
        if l > self.k :
            for i in xrange(0, l - self.k) :
                self.total += 1
                key = data[i:i + self.k]
                counter = None
                if key in self.counters : 
                    counter = self.counters[key]
                else :
                    counter = Counter()
                    self.counters[key] = counter
                counter[data[i + self.k]] += 1

class BayesSequenceClassifier(object):
    
    def __init__(self, k = 3):
        self.k = k
        self.models = {}
        self.classifier_dict = {}
        
    
    def train(self, data, label):
        model = None
        if label not in self.models :
            model = BayesModel(self.k)
            self.models[label] = model
        else :
            model = self.models[label]
        model.train(data)
    
    def __probability_array(self, data):
        l = len(data)
        ls = []
        if l > self.k :
            for i in xrange(0, l - self.k) :
                prior = data[i:i + self.k]
                result = data[i + self.k]
                e = (prior, result)
                if e in self.classifier_dict :
                    ls.append(self.classifier_dict[e])
        return ls
                    
                    
    def prepare(self):
        base_dict = {}
        all_priors_and_results = Set()
        for label in self.models :
                base_dict[label] = {}
                model = self.models[label]
                for prior in model.counters :
                    total = 0
                    counts = model.counters[prior]
                    for result in counts :
                        all_priors_and_results.add((prior, result))
                        total += counts[result]
                    for result in counts :
                        base_dict[label][(prior, result)] = (counts[result] + 0.0) / total
        self.classifier_dict = {}
        
        for entry in all_priors_and_results :
            max_label = None
            max_value = 0.0
            for label in base_dict :
                if not entry in base_dict[label] :
                    continue
                prob = base_dict[label][entry]
                if (prob > max_value) :
                    max_value = prob
                    max_label = label
            if max_label is not None :
                self.classifier_dict[entry] = max_label
                
    def classify(self, data):
        probs = self.__probability_array(data)
        result = Counter(probs).most_common(1)
        if len(result) == 0 :
            return "NONE"
        return result[0][0]

def test_bayes_model():
    ls = ["AGTGCAGTT"]
    model = BayesModel()
    for s in ls :
        model.train(s)
    with_t = model.counters['AGT']['T'] == 1
    return model.counters['AGT']['G'] == 1 and with_t

def test_bayes_classifier():
    classifier = BayesSequenceClassifier()
    classifier.train("0001", "A")
    classifier.train("0000", "B")
    classifier.train("00010000", "C")
    classifier.prepare()
    return classifier.classify("0000000000000001")

print test_bayes_model()
print test_bayes_classifier()
        
def run_on_sample():
    pass

class NaiveBayesRun():
    
    def __init__(self, classifier, train_file, test_file):
        self.classifier = classifier
        self.test_file = test_file
        self.train_file = train_file
        
    def run(self):
        with open(self.train_file, "r") as f:
            for l in f.readlines() :
                splitted = l.split("###")
                self.classifier.train(splitted[0], splitted[1])
        
        self.classifier.prepare()
        success = Counter()
        with open(self.test_file, "r") as f :
            for l in f.readlines() :
                splitted = l.split("###")
                print self.classifier.classify(splitted[0])
                success[self.classifier.classify(splitted[0]) == splitted[1]] += 1
        return success
                
def sample_run():
    run = NaiveBayesRun(BayesSequenceClassifier(k = 12), "train_sample", "test_sample")
    print run.run()
        
        
sample_run()
    