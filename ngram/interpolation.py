'''
@author: bighouse
'''
from _collections import defaultdict
from ngram import NgramSequenceClassifier, Classifier, choose_max

class InterpolationClassifier(Classifier):
    
    def __init__(self, n, weights, verbose = False, cached = False):
        self.cached = cached
        if self.cached :
            self.cache = {}
        
        self.weights = weights
        self.n = n
        self.verbose = verbose
        self.k = len(weights)
        self.classifiers = []
        
        for i in range(self.k) :
            self.classifiers.append(NgramSequenceClassifier(n - i))
        
    def train(self, data, label):
        for i in xrange(self.k) :
            self.classifiers[i].train(data[i:], label)
        
    def prepare(self):
        pass

    def get_prob(self, denominator, classifier, label, current_ngram):
        model = classifier.models[label]
        numerator = classifier.get_prior_probabability_numerator(model, current_ngram[:-1])
        prior = numerator / denominator
        return model.conditional_probability(current_ngram) * prior
    
    def add_terms_of_submodel(self, probas, current_ngram, classifier, weight) :
            denominator = classifier.get_prior_probabability_denominator(current_ngram[:-1])
                # If the prior has never been seen
            if denominator == 0 :
                return 0.0
            else:
                for label in classifier.models :
                    frequency = self.get_prob(denominator, classifier, label, current_ngram)
                    probas[label] +=  frequency * weight
    
    def classify_suffix(self, probas, data, idx):
        for j in xrange(self.k):
            weight = self.weights[j]
            classifier = self.classifiers[j]
            subsequence = data[idx - self.n + j : idx + 1]
            self.add_terms_of_submodel(probas, subsequence, classifier, weight)
        return probas
    
    def classify(self, data):
        d_length = len(data)
        if d_length < self.n + 1 :
            return 'NONE'
        predictions = []
        for i in xrange(self.n, d_length) :
            probas = defaultdict(lambda : 0.0)
            probas = self.classify_suffix(probas, data, i)
            predictions.append(probas)
        result = choose_max(predictions)
        if len(result) == 0 :
            return "NONE"
        return result[0][0]
        