'''
@author: bighouse
'''
from ngram import NgramSequenceClassifier
from ngram import Classifier
from _collections import defaultdict
from collections import Counter
from numpy.core.test_rational import numerator

class __TEMPLATE(Classifier):
    
    def __init__(self, n, verbose = False, cached = False):
        self.cached = cached
        if self.cached :
            self.cache = {}
        
        self.n = n
        self.verbose = verbose
        self.classifier = NgramSequenceClassifier(self.n)
        
    def train(self, data, label):
        for i in xrange(self.k) :
            self.classifiers[i].train(data[i:], label)
        
    def prepare(self):
        pass

    def __select_max(self, probas):
        max_label, max_value = 'NONE', 0.0
        for key, value in probas.iteritems() :
            if value > max_value :
                max_value = value
                max_label = key
        return max_label, max_value
        
    
    def classify(self, data):
        d_length = len(data)
        if d_length < self.n :
            return 'NONE'
        predictions = []
        
        for i in xrange(self.n, d_length - 1) :
            probas = defaultdict(lambda : 0.0)
            classifier = self.classifier
            subsequence = data[i - self.n : i + 1]
            denominator = classifier.get_prior_probabability_denominator(subsequence[:-1])
            if self.cached and subsequence in self.cache :
                best = self.cache[subsequence]
            else :
                for label in classifier.models :
                    numerator = classifier.get_prior_probabability_numerator(label, subsequence[:-1])
                    prior = numerator / denominator
                    model = classifier.models[label]
                    classifier.models[label]
                    probas[label] += model.conditional_probability(subsequence)
                best = self.__select_max(probas)
                if self.cached :
                    self.cache[data] = best
            if best[1] > 1 or self.verbose :
                print data, best
            predictions.append(best)
        result = Counter(map(lambda x : x[0], predictions)).most_common(1)
        if len(result) == 0 :
            return "NONE"
        return result[0][0]
        