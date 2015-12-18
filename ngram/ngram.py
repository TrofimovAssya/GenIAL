'''
@author: bighouse
'''

from sets import Set
from collections import Counter
from math import log, exp

class Constants :
        UNKNOWN = "N"

def choose(predictions):
    total = {}
    VERY_SMALL = -100000
    for probas in predictions :
        for label, proba in probas.iteritems():
            try :
                if proba == 0 :
                    proba = VERY_SMALL
                if label not in total :
                    total[label] = log(proba)
                else :
                    total[label] += log(proba)
            except Exception :
                total[label] = VERY_SMALL
    max_label = None
    max_value = float("-inf")
    for label, value in total.iteritems() :
        if max_value < value :
            max_value = value
            max_label = label
    return max_label

def __select_max(probas):
    max_label, max_value = 'NONE', 0.0
    for key, value in probas.iteritems() :
        if value > max_value :
            max_value = value
            max_label = key
    return max_label, max_value

def choose_max(predictions):
    tally = []
    for probas in predictions :
        tally.append(__select_max(probas)[0])
    max_label = Counter(map(lambda x : x[0], tally)).most_common(1)
    return max_label
    
# Proof of concept
# This algorithm is really naive, but may work an alphabet of size 4.
class FrequencyModel(object) :
    
    def __init__(self, k = 3):
        self.k = k
        self.counters = {}
        self.total = 0
        
    def probability(self, observation):
        prior = observation[:-1]
        posterior = observation[-1]
        if prior in self.counters :
            counter = self.counters[prior]
            if posterior in counter :
                return counter[posterior] / self.total
        else :
            return 0.0
    
    def conditional_probability(self, observation):
        prior = observation[:-1]
        posterior = observation[-1]
        
        if prior in self.counters :
            counter = self.counters[prior]
            accu = 0.0
            posterior_proba = 0.0
            for val in counter :
                count = counter[val]
                accu += count
                if val == posterior :
                    posterior_proba = count
            return posterior_proba / accu
        else :
            return 0.0
        
    def train(self, data):
        l = len(data)
        if l > self.k :
            for i in xrange(0, l - self.k) :
                self.total += 1
                prior = data[i:i + self.k] # This is the prior
                posterior = data[i + self.k]
                if posterior == Constants.UNKNOWN :
                    continue
                if Constants.UNKNOWN in posterior :
                    continue
                
                counter = None
                # Initialize
                if prior in self.counters : 
                    counter = self.counters[prior]
                else :
                    counter = Counter()
                    self.counters[prior] = counter
                    
                counter[posterior] += 1

class Classifier(object):
    
    def train(self, data, label):
        raise NotImplementedError()
    
    def prepare(self):
        raise NotImplementedError()
    
    def classify(self, data):
        raise NotImplementedError()

class NgramSequenceClassifier(Classifier):
    
    def __init__(self, k = 3):
        self.k = k
        self.models = {}
        self.classifier_dict = {}
        self.count_classifications = 0
    
    def train(self, data, label):
        model = None
        if label not in self.models :
            model = FrequencyModel(self.k)
            self.models[label] = model
        else :
            model = self.models[label]
        model.train(data)
    
    # Gives the predicted result for each character
    def __probability_array(self, data):
        l = len(data)
        ls = []
        if l > self.k :
            for i in xrange(0, l - self.k) :
                prior = data[i : i + self.k]
                result = data[i + self.k]
                e = (prior, result)
                if e in self.classifier_dict :
                    ls.append(self.classifier_dict[e])
        return ls
    
    def get_prior_probabability_denominator(self, prior):
        accu = 0.0
        for label, model in self.models.iteritems():
            if prior in model.counters :
                for letter, val in model.counters[prior].iteritems():
                    accu += val
        return accu
    
    def get_prior_probabability_numerator(self, model, prior):
        accu = 0.0
        if prior not in model.counters :
            return 0
        for letter, val in model.counters[prior].iteritems():
            accu += val
        return accu
        
    #In a first step, normalize counts into probabilities. After, build a cache holding the decisions for each n-gram.                    
    def prepare(self):
        prior_posterior_probs = {}
        prior_counter = {}
        all_priors_counter = Counter()
        all_priors_and_posteriors = Set()
        #For each model, compute the probability that the posterior follows the prior
        for label in self.models :
                
                prior_counter_for_label = {}
                prior_posterior_probs[label] = {}
                prior_counter[label] = prior_counter_for_label
                
                model = self.models[label]
                
                for prior in model.counters :
                    total = 0
                    counts = model.counters[prior]
                    # Get totals
                    for posterior in counts :
                        total += counts[posterior]
                    prior_counter_for_label[prior] = total
                    all_priors_counter[prior] += total
                    # Compute the sample probability for the posterior for the given model
                    for posterior in counts :
                        prior_posterior_probs[label][(prior, posterior)] = (counts[posterior] + 0.0) / total
                        all_priors_and_posteriors.add((prior, posterior))
                    
        self.classifier_dict = {}
        
        #Select the class that would classify this n-gram.
        for entry in all_priors_and_posteriors :
            prior = entry[0]
            max_label = None
            max_probability = 0.0
            for label in prior_posterior_probs :
                
                if not entry in prior_posterior_probs[label] :
                    #This is where backoff could occur
                    continue
                prior_probability = ((prior_counter[label][prior] + 0.0) / all_priors_counter[prior])
                prob = prior_posterior_probs[label][entry] * prior_probability
                if (prob > max_probability) :
                    max_probability = prob
                    max_label = label
            if max_label is not None :
                self.classifier_dict[entry] = (max_label, max_probability)
    
    
    
    def classify(self, data):
        probs = self.__probability_array(data)
        self.count_classifications += 1
        result = Counter(map(lambda x : x[0], probs)).most_common(1)
        if len(result) == 0 :
            return "NONE"
        return result[0][0]
