'''
@author: bighouse
'''
from ngram import NgramSequenceClassifier
from ngram import Classifier
from ngram import choose
from _collections import defaultdict

class AddDeltaClassifier(Classifier):
    
    def __init__(self, n, vocab_size, verbose = False, cached = False, delta = 1.0):
        self.cached = cached
        if self.cached :
            self.cache = {}
        self.vocab_size = vocab_size
        self.n = n
        self.verbose = verbose
        self.classifier = NgramSequenceClassifier(self.n)
        self.delta = delta
        
    def train(self, data, label):
        self.classifier.train(data, label)
        
    def prepare(self):
        pass

    def __select_max(self, probas):
        max_label, max_value = 'NONE', 0.0
        for key, value in probas.iteritems() :
            if value > max_value :
                max_value = value
                max_label = key
        return max_label, max_value
    
    # Retourne la probabilite lissee du n-1 gramme 
    def weighted_prob(self, model, observation):
        new_total = self.vocab_size ** self.n
        
        prior = observation[:-1]
        posterior = observation[-1]
        numer = 0.0
        if prior in model.counters:
            counter = model.counters[prior]
            denom = 0.0
            
            for val in counter :
                count = counter[val]
                denom += count
                if val == posterior :
                    numer = count
            # ADD DELTA
            numer += self.delta
            denom += self.vocab_size * self.delta
            return numer / denom
        else :
            return self.delta / new_total
    
    def get_prob(self, denominator, classifier, label, current_ngram):
        model = classifier.models[label]
        numerator = classifier.get_prior_probabability_numerator(model, current_ngram[:-1])
        #Because of smoothing
        numerator += self.delta
        prior = numerator / denominator
        proba = self.weighted_prob(model, current_ngram)
        result = proba * prior
        return result
    
    def classify_suffix(self, classifier, current_ngram):
            probas = defaultdict(lambda : 0.0)
            denominator = classifier.get_prior_probabability_denominator(current_ngram[:-1]) + self.delta * self.vocab_size
            # For laplace smoothing, there ain't nothing like an unobserved sequence!
                # Each model is smoothed, so you have to 
                # Get the probability for the model
            for label in classifier.models :
                probas[label] = self.get_prob(denominator, classifier, label, current_ngram)
            return probas
    
    
    # A sequence is classified by classying its suffixes
    def classify(self, data):
        d_length = len(data)
        if d_length < self.n + 1 :
            return 'NONE'
        predictions = []
        for i in xrange(self.n, d_length) :
            classifier = self.classifier
            current_ngram = data[i - self.n : i + 1]
            probas = self.classify_suffix(classifier, current_ngram)
            predictions.append(probas)
        result = choose(predictions)
        if len(result) == 0 :
            return "NONE"
        return result[0][0]
        