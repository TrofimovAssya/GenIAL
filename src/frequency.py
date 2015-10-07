'''
Created on Oct 4, 2015

@author: bighouse
'''

from sets import Set
from collections import Counter


# Proof of concept
# This algorithm is really naive, but may work an alphabet of size 4.
class FrequencyModel(object) :
    
    def __init__(self, k = 3):
        self.k = k
        self.counters = {}
        self.total = 0
        
    def train(self, data):
        l = len(data)
        if l > self.k :
            for i in xrange(0, l - self.k) :
                self.total += 1
                prior = data[i:i + self.k] # This is the prior
                posterior = data[i + self.k]
                counter = None
                # Initialize
                if prior in self.counters : 
                    counter = self.counters[prior]
                else :
                    counter = Counter()
                    self.counters[prior] = counter
                counter[posterior] += 1

class KgramSequenceClassifier(object):
    
    def __init__(self, k = 3):
        self.k = k
        self.models = {}
        self.classifier_dict = {}
    
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
        
        for entry in all_priors_and_posteriors :
            prior = entry[0]
            max_label = None
            max_value = 0.0
            for label in prior_posterior_probs :
                if not entry in prior_posterior_probs[label] :
                    continue
                prior_probability = ((prior_counter[label][prior] + 0.0) / all_priors_counter[prior])
                prob = prior_posterior_probs[label][entry] * prior_probability
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
    model = FrequencyModel()
    for s in ls :
        model.train(s)
    with_t = model.counters['AGT']['T'] == 1
    return model.counters['AGT']['G'] == 1 and with_t

def test_kgram_classifier():
    classifier = KgramSequenceClassifier()
    classifier.train("0001", "A")
    classifier.train("0000", "B")
    classifier.train("00010000", "C")
    classifier.prepare()
    return classifier.classify("0000000000000001")


def tests():
    print test_bayes_model()
    print test_kgram_classifier()
tests()


class FrequencyRun():
    
    def __init__(self, classifier, train_file, test_file, validation_file):
        self.classifier = classifier
        self.test_file = test_file
        self.validation_file = validation_file
        self.train_file = train_file
        
    def validate(self):
        with open(self.train_file, "r") as f:
            for l in f.readlines() :
                splitted = l.split("###")
                self.classifier.train(splitted[0], splitted[1][:-1])
        
        self.classifier.prepare()
        return self.classify(self.validation_file)
        
    
    def classify(self, file_to_classify):
        confusion_matrix = Counter()
        with open(file_to_classify, "r") as f :
            for l in f.readlines() :
                splitted = l.split("###")
                predicted = self.classifier.classify(splitted[0])
                actual = splitted[1][:-1]
                confusion_matrix[(predicted, actual)] += 1
        return confusion_matrix
    
    def test(self):
        return self.validate(self.test_file)
                
def bayes_sample_run():
    '''
    For k =  13
    Success rate  0.612236357549
    Counter({('chr1', 'chr1'): 1349, ('chr19', 'chr19'): 1045, ('chr6', 'chr6'): 1036, ('chr12', 'chr12'): 998, ('chrM', 'chrM'): 967, ('chr16', 'chr16'): 932, ('chr17', 'chr17'): 802, ('chr11', 'chr11'): 790, ('chr9', 'chr9'): 746, ('chr7', 'chr7'): 705, ('chr2', 'chr2'): 697, ('chr3', 'chr3'): 623, ('chr5', 'chr5'): 566, ('chr15', 'chr15'): 432, ('chrX', 'chrX'): 430, ('chr22', 'chr22'): 385, ('chr8', 'chr8'): 362, ('chr4', 'chr4'): 339, ('chr10', 'chr10'): 332, ('chr14', 'chr14'): 280, ('chr20', 'chr20'): 242, ('chr13', 'chr13'): 233, ('chr6', 'chr7'): 148, ('chr7', 'chr6'): 125, ('chr21', 'chr21'): 113, ('chr18', 'chr18'): 109, ('chrUn_gl000220', 'chrUn_gl000220'): 90, ('chr1', 'chr17'): 90, ('chr12', 'chr5'): 88, ('chr12', 'chr7'): 83, ('chr6', 'chr5'): 79, ('chr1', 'chr2'): 79, ('chrM', 'chr1'): 78, ('chr3', 'chr1'): 73, ('chr1', 'chr3'): 68, ('chr2', 'chr1'): 66, ('chr7', 'chr1'): 66, ('chr9', 'chr1'): 66, ('chr1', 'chr16'): 65, ('chr1', 'chr5'): 65, ('chr1', 'chr12'): 64, ('chr1', 'chr19'): 59, ('chr12', 'chr1'): 58, ('chr19', 'chr1'): 56, ('chr17', 'chr1'): 55, ('chr12', 'chr16'): 55, ('chr1', 'chr11'): 53, ('chr16', 'chr17'): 53, ('chr12', 'chr2'): 53, ('chr1', 'chr7'): 53, ('chr3', 'chr2'): 51, ('chr11', 'chr1'): 50, ('chr12', 'chr20'): 49, ('chr4', 'chr15'): 48, ('chr1', 'chr6'): 48, ('chr3', 'chr19'): 48, ('chr7', 'chr11'): 47, ('chr12', 'chr3'): 47, ('chr7', 'chr2'): 47, ('chr16', 'chr1'): 46, ('chr12', 'chr6'): 46, ('chr7', 'chr5'): 46, ('chr6', 'chr1'): 45, ('chr1', 'chr14'): 44, ('chr16', 'chr19'): 44, ('chr3', 'chr11'): 44, ('chr1', 'chr10'): 44, ('chr12', 'chr19'): 43, ('chr12', 'chr17'): 43, ('chr7', 'chr12'): 42, ('chr6', 'chr2'): 42, ('chr7', 'chr3'): 41, ('chr17', 'chr12'): 41, ('chr3', 'chr12'): 40, ('chr7', 'chr17'): 39, ('chr2', 'chr17'): 39, ('chr1', 'chr4'): 38, ('chr19', 'chr2'): 38, ('chr12', 'chr22'): 38, ('chr1', 'chr8'): 38, ('chr19', 'chr17'): 37, ('chr5', 'chr2'): 37, ('chr7', 'chr19'): 37, ('chr19', 'chr12'): 36, ('chr2', 'chr6'): 36, ('chr2', 'chr12'): 36, ('chr11', 'chr2'): 36, ('chr1', 'chr9'): 36, ('chr3', 'chr7'): 35, ('chr16', 'chr11'): 35, ('chr1', 'chrX'): 35, ('chr19', 'chr10'): 35, ('chr19', 'chr7'): 35, ('chr3', 'chr5'): 34, ('chr17', 'chr16'): 34, ('chr11', 'chr19'): 34, ('chr17', 'chr2'): 33, ('chr11', 'chr17'): 33, ('chr7', 'chr16'): 33, ('chrX', 'chrY'): 33, ('chr19', 'chr3'): 33, ('chr6', 'chr3'): 32, ('chr11', 'chr7'): 32, ('chr2', 'chr5'): 32, ('chr16', 'chr12'): 32, ('chr6', 'chr17'): 32, ('chr11', 'chr3'): 32, ('chr6', 'chr11'): 31, ('chr7', 'chr4'): 31, ('chr12', 'chr11'): 30, ('chr1', 'chr20'): 30, ('chr12', 'chr9'): 30, ('chr5', 'chr1'): 30, ('chr3', 'chr17'): 30, ('chr19', 'chr16'): 30, ('chr3', 'chr4'): 30, ('chr3', 'chr8'): 29, ('chr2', 'chr15'): 29, ('chrX', 'chr1'): 29, ('chr5', 'chr6'): 29, ('chr3', 'chr9'): 29, ('chr6', 'chr12'): 29, ('chr17', 'chr19'): 28, ('chr11', 'chrX'): 28, ('chr19', 'chr11'): 28, ('chr5', 'chr17'): 28, ('chr11', 'chr5'): 28, ('chrY', 'chrY'): 27, ('chr17', 'chr6'): 27, ('chr1', 'chr15'): 27, ('chr17', 'chr3'): 27, ('chr11', 'chr6'): 27, ('chr5', 'chr7'): 27, ('chr12', 'chr4'): 26, ('chr3', 'chr16'): 26, ('chr7', 'chr22'): 26, ('chr3', 'chr14'): 26, ('chr10', 'chr2'): 26, ('chr17', 'chr11'): 26, ('chr8', 'chr1'): 26, ('chr3', 'chr6'): 26, ('chr2', 'chr13'): 26, ('chr4', 'chrX'): 25, ('chr1', 'chrM'): 25, ('chr6', 'chr16'): 25, ('chr1', 'chr13'): 25, ('chr12', 'chrX'): 25, ('chr16', 'chr7'): 25, ('chr17', 'chr9'): 25, ('chrX', 'chr2'): 25, ('chr19', 'chr5'): 25, ('chr2', 'chr9'): 24, ('chr4', 'chr10'): 24, ('chr19', 'chr8'): 24, ('chrX', 'chr3'): 24, ('chr9', 'chr2'): 24, ('chr5', 'chr11'): 24, ('chr19', 'chr6'): 24, ('chr1', 'chr22'): 24, ('chr4', 'chr2'): 24, ('chr19', 'chr9'): 24, ('chr3', 'chr15'): 24, ('chr19', 'chr15'): 24, ('chr16', 'chr2'): 23, ('chr2', 'chr19'): 23, ('chr7', 'chrX'): 23, ('chr17', 'chr5'): 23, ('chr2', 'chrX'): 23, ('chr5', 'chr3'): 23, ('chr16', 'chrX'): 22, ('chr16', 'chr3'): 22, ('chr3', 'chrX'): 22, ('chr10', 'chr16'): 22, ('chr7', 'chr9'): 22, ('chr12', 'chr8'): 22, ('chr2', 'chr7'): 22, ('chr3', 'chr10'): 22, ('chr16', 'chr10'): 22, ('chr6', 'chr15'): 22, ('chr9', 'chr3'): 22, ('chr2', 'chr16'): 21, ('chr17', 'chr22'): 21, ('chr6', 'chr20'): 21, ('chr7', 'chr8'): 21, ('chr19', 'chrX'): 21, ('chr5', 'chr12'): 21, ('chr16', 'chr9'): 21, ('chr7', 'chr10'): 21, ('chr10', 'chr7'): 21, ('chr9', 'chr7'): 21, ('chr17', 'chr10'): 20, ('chr15', 'chr1'): 20, ('chr11', 'chr12'): 20, ('chrY', 'chrX'): 20, ('chr4', 'chr3'): 20, ('chr12', 'chr15'): 20, ('chr11', 'chr8'): 20, ('chrX', 'chr19'): 20, ('chr9', 'chr17'): 20, ('chr4', 'chr1'): 20, ('chr11', 'chr4'): 20, ('chr2', 'chr4'): 20, ('chr11', 'chr16'): 20, ('chr5', 'chr16'): 20, ('chrX', 'chr6'): 19, ('chr10', 'chr1'): 19, ('chr13', 'chr1'): 19, ('chr5', 'chr4'): 19, ('chr11', 'chr9'): 19, ('chr6', 'chr19'): 19, ('chr22', 'chr1'): 19, ('chr2', 'chr11'): 19, ('chr14', 'chr1'): 19, ('chr15', 'chr2'): 18, ('chr2', 'chr3'): 18, ('chr10', 'chr4'): 18, ('chr7', 'chr15'): 18, ('chr5', 'chr8'): 18, ('chr6', 'chrX'): 18, ('chr19', 'chr14'): 18, ('chr13', 'chr7'): 18, ('chr5', 'chr10'): 18, ('chr10', 'chr9'): 18, ('chr6', 'chr4'): 18, ('chr3', 'chr13'): 17, ('chr22', 'chr17'): 17, ('chr17', 'chr15'): 17, ('chr16', 'chr4'): 17, ('chr2', 'chr10'): 17, ('chr5', 'chr15'): 17, ('chr16', 'chr5'): 17, ('chr7', 'chr14'): 17, ('chr17', 'chr8'): 16, ('chr12', 'chr13'): 16, ('chr9', 'chr6'): 16, ('chr6', 'chr9'): 16, ('chr16', 'chr15'): 16, ('chr17', 'chr7'): 16, ('chr12', 'chr10'): 16, ('chr9', 'chr12'): 16, ('chr11', 'chr14'): 16, ('chr16', 'chr8'): 16, ('chr2', 'chr14'): 16, ('chr16', 'chr6'): 16, ('chr1', 'chr18'): 15, ('chr19', 'chr22'): 15, ('chr11', 'chr15'): 15, ('chr3', 'chr20'): 15, ('chr10', 'chr8'): 15, ('chr11', 'chr22'): 15, ('chr17', 'chrX'): 15, ('chr22', 'chr11'): 15, ('chr2', 'chr20'): 15, ('chr5', 'chr14'): 15, ('chr6', 'chr8'): 15, ('chr5', 'chr9'): 15, ('chr5', 'chrX'): 15, ('chr4', 'chr17'): 15, ('chr19', 'chr20'): 15, ('chr15', 'chr9'): 14, ('chr10', 'chr11'): 14, ('chr4', 'chr5'): 14, ('chr10', 'chr17'): 14, ('chr19', 'chr13'): 14, ('chr16', 'chr20'): 14, ('chr14', 'chr6'): 14, ('chr9', 'chr4'): 14, ('chr16', 'chr22'): 14, ('chr6', 'chr14'): 14, ('chr9', 'chr11'): 13, ('chr22', 'chr19'): 13, ('chr6', 'chr13'): 13, ('chr19', 'chr4'): 13, ('chr15', 'chr17'): 13, ('chr10', 'chr19'): 13, ('chr5', 'chr13'): 13, ('chr17', 'chr14'): 13, ('chr12', 'chr14'): 13, ('chr10', 'chrX'): 13, ('chr16', 'chr14'): 13, ('chr20', 'chr1'): 12, ('chr4', 'chr12'): 12, ('chr17', 'chr20'): 12, ('chrX', 'chr5'): 12, ('chr6', 'chr22'): 12, ('chr13', 'chr5'): 12, ('chr7', 'chr21'): 12, ('chr11', 'chr18'): 12, ('chr6', 'chr21'): 12, ('chr6', 'chr10'): 12, ('chr15', 'chr7'): 12, ('chr2', 'chr22'): 12, ('chr9', 'chr19'): 12, ('chr4', 'chr6'): 12, ('chr7', 'chr20'): 12, ('chr13', 'chr2'): 12, ('chr9', 'chr14'): 12, ('chr5', 'chr19'): 12, ('chr11', 'chr10'): 11, ('chr10', 'chr3'): 11, ('chr5', 'chr20'): 11, ('chrX', 'chr10'): 11, ('chr8', 'chr3'): 11, ('chr10', 'chr12'): 11, ('chr4', 'chr16'): 11, ('chr18', 'chr19'): 11, ('chr13', 'chr12'): 11, ('chr22', 'chr3'): 11, ('chr9', 'chr22'): 11, ('chrX', 'chr8'): 11, ('chr9', 'chr10'): 11, ('chrX', 'chr7'): 11, ('chr15', 'chr11'): 11, ('chr14', 'chr22'): 11, ('chr15', 'chr16'): 11, ('chr10', 'chr14'): 11, ('chr22', 'chr14'): 11, ('chr13', 'chr4'): 11, ('chr15', 'chr3'): 11, ('chr7', 'chr13'): 11, ('chr9', 'chr16'): 11, ('chrX', 'chr12'): 10, ('chr9', 'chr5'): 10, ('chr18', 'chr3'): 10, ('chr4', 'chr19'): 10, ('chr20', 'chr6'): 10, ('chr2', 'chr8'): 10, ('chr4', 'chr13'): 10, ('chrX', 'chr16'): 10, ('chr17', 'chr21'): 10, ('chrX', 'chr11'): 10, ('chr4', 'chr11'): 10, ('chr15', 'chr14'): 10, ('chr8', 'chr5'): 10, ('chr15', 'chr12'): 10, ('chr4', 'chr7'): 9, ('chr4', 'chr9'): 9, ('chr18', 'chr15'): 9, ('chr2', 'chr21'): 9, ('chr15', 'chr19'): 9, ('chr15', 'chr4'): 9, ('chr10', 'chr15'): 9, ('chr8', 'chr7'): 9, ('chr19', 'chr21'): 9, ('chr15', 'chrX'): 9, ('chr11', 'chr13'): 9, ('chr5', 'chr21'): 9, ('chr8', 'chr16'): 9, ('chr20', 'chr2'): 9, ('chr8', 'chr14'): 9, ('chr12', 'chr18'): 9, ('chr15', 'chr20'): 9, ('chr20', 'chr7'): 9, ('chr10', 'chr5'): 9, ('chr14', 'chr15'): 9, ('chr14', 'chr10'): 8, ('chr7', 'chr18'): 8, ('chr1', 'chr21'): 8, ('chr11', 'chr20'): 8, ('chr10', 'chr6'): 8, ('chr13', 'chr16'): 8, ('chr8', 'chr9'): 8, ('chr3', 'chr22'): 8, ('chr8', 'chr17'): 8, ('chr18', 'chr12'): 8, ('chrX', 'chr17'): 8, ('chr8', 'chr4'): 8, ('chr9', 'chr20'): 8, ('chr8', 'chr15'): 8, ('chr14', 'chr18'): 8, ('chrX', 'chr15'): 8, ('chr20', 'chr17'): 8, ('chr19', 'chr18'): 8, ('chr8', 'chr2'): 8, ('chr9', 'chr15'): 8, ('chr18', 'chr1'): 8, ('chr15', 'chr6'): 8, ('chr14', 'chr11'): 8, ('chrX', 'chr14'): 8, ('chr14', 'chr5'): 7, ('chr17', 'chr4'): 7, ('chr13', 'chr19'): 7, ('chr13', 'chr8'): 7, ('chr3', 'chr18'): 7, ('chrX', 'chr20'): 7, ('chr8', 'chrX'): 7, ('chrX', 'chr13'): 7, ('NONE', 'chr1'): 7, ('chr14', 'chr4'): 7, ('chr13', 'chr17'): 7, ('chr15', 'chr5'): 7, ('chr6', 'chr18'): 7, ('chr8', 'chr6'): 7, ('chr22', 'chr12'): 7, ('chr22', 'chr2'): 7, ('chr20', 'chr16'): 6, ('chr5', 'chr22'): 6, ('chr20', 'chr3'): 6, ('chr18', 'chr5'): 6, ('chr10', 'chr20'): 6, ('chr21', 'chr17'): 6, ('chr14', 'chr9'): 6, ('chr14', 'chr12'): 6, ('chr13', 'chr3'): 6, ('chr4', 'chr8'): 6, ('chr22', 'chr16'): 6, ('chr22', 'chr7'): 6, ('chr14', 'chr3'): 6, ('chr1', 'chrY'): 6, ('chr18', 'chr2'): 6, ('chr8', 'chr11'): 6, ('chr8', 'chr12'): 6, ('chr4', 'chr20'): 5, ('chr15', 'chr8'): 5, ('chr21', 'chr16'): 5, ('chrX', 'chr9'): 5, ('chr9', 'chrX'): 5, ('chr14', 'chr2'): 5, ('chrX', 'chr4'): 5, ('chr14', 'chr7'): 5, ('chr17', 'chr13'): 5, ('chr14', 'chrX'): 5, ('chrY', 'chr15'): 5, ('chr20', 'chr19'): 5, ('chr22', 'chrX'): 5, ('chr22', 'chr4'): 5, ('chrY', 'chr2'): 5, ('chr18', 'chr6'): 5, ('chr4', 'chr14'): 5, ('chr13', 'chr10'): 5, ('chr13', 'chr15'): 5, ('chr11', 'chr21'): 5, ('chr14', 'chr8'): 5, ('chr16', 'chr13'): 5, ('chr22', 'chr8'): 5, ('chr15', 'chr22'): 5, ('chr13', 'chr11'): 5, ('chr4', 'chr18'): 5, ('chr14', 'chr17'): 5, ('chr20', 'chr11'): 4, ('chr10', 'chr22'): 4, ('chr15', 'chr13'): 4, ('chr8', 'chr10'): 4, ('chr22', 'chr10'): 4, ('chr22', 'chr15'): 4, ('chr22', 'chr6'): 4, ('chr20', 'chr15'): 4, ('chr18', 'chr20'): 4, ('chr15', 'chr10'): 4, ('chr5', 'chr18'): 4, ('chr14', 'chr16'): 4, ('chr18', 'chr16'): 4, ('chr16', 'chr18'): 4, ('chr20', 'chr10'): 4, ('chr3', 'chr21'): 4, ('chr17', 'chr18'): 4, ('chr22', 'chr21'): 4, ('chr20', 'chr5'): 4, ('chr2', 'chr18'): 4, ('chr13', 'chr9'): 4, ('chr15', 'chr18'): 4, ('chr20', 'chr14'): 4, ('chr21', 'chr12'): 4, ('chr20', 'chr8'): 4, ('chr18', 'chr8'): 4, ('chr14', 'chr19'): 3, ('chr3', 'chrY'): 3, ('chr21', 'chrX'): 3, ('chr9', 'chr8'): 3, ('chr21', 'chr19'): 3, ('chr10', 'chr18'): 3, ('chr20', 'chr21'): 3, ('chr22', 'chr9'): 3, ('chr15', 'chr21'): 3, ('chr20', 'chr22'): 3, ('NONE', 'chrX'): 3, ('chrX', 'chr22'): 3, ('chr20', 'chr4'): 3, ('chr8', 'chr19'): 3, ('chrY', 'chr9'): 3, ('chr21', 'chr6'): 3, ('NONE', 'chr2'): 3, ('chr18', 'chr10'): 3, ('chr20', 'chr9'): 3, ('chr18', 'chr9'): 3, ('chr21', 'chr18'): 3, ('chr12', 'chr21'): 3, ('chr8', 'chr20'): 3, ('chr18', 'chr4'): 3, ('chr21', 'chr3'): 3, ('chr10', 'chr21'): 3, ('chr16', 'chrY'): 3, ('chr18', 'chrX'): 3, ('chr21', 'chr11'): 3, ('chr8', 'chr18'): 3, ('chr8', 'chr22'): 3, ('chrX', 'chr18'): 3, ('chrY', 'chr12'): 3, ('chr18', 'chr13'): 3, ('chr13', 'chr14'): 3, ('chrX', 'chr21'): 3, ('chr13', 'chr20'): 3, ('chr16', 'chr21'): 3, ('chr14', 'chr20'): 3, ('chr9', 'chr21'): 3, ('chr9', 'chr13'): 3, ('NONE', 'chr22'): 2, ('chr21', 'chr2'): 2, ('chr6', 'chrY'): 2, ('chrY', 'chr11'): 2, ('chr22', 'chr20'): 2, ('chr21', 'chr5'): 2, ('NONE', 'chr7'): 2, ('chr18', 'chr17'): 2, ('chrY', 'chr20'): 2, ('chr9', 'chr18'): 2, ('NONE', 'chr17'): 2, ('chr18', 'chr22'): 2, ('chr14', 'chr21'): 2, ('chrUn_gl000220', 'chr11'): 2, ('NONE', 'chr18'): 2, ('chr21', 'chr15'): 2, ('chr13', 'chr6'): 2, ('chr13', 'chr21'): 2, ('chr14', 'chr13'): 2, ('chr4', 'chr21'): 2, ('chr21', 'chr4'): 2, ('chr21', 'chr13'): 2, ('chrUn_gl000220', 'chr21'): 2, ('NONE', 'chr20'): 2, ('NONE', 'chr6'): 2, ('chrY', 'chr19'): 2, ('chr18', 'chr14'): 2, ('chr22', 'chr18'): 2, ('chr20', 'chr12'): 2, ('chrUn_gl000220', 'chr12'): 2, ('NONE', 'chr19'): 2, ('chr21', 'chr1'): 2, ('NONE', 'chr3'): 2, ('chrY', 'chr3'): 2, ('chr4', 'chr22'): 2, ('chr10', 'chr13'): 2, ('chr13', 'chr18'): 2, ('NONE', 'chr10'): 2, ('chr20', 'chr18'): 2, ('chr18', 'chr11'): 2, ('chrY', 'chr4'): 1, ('chrM', 'chr22'): 1, ('chr18', 'chr7'): 1, ('chr7', 'chrY'): 1, ('NONE', 'chr12'): 1, ('chr21', 'chr10'): 1, ('chrX', 'chrUn_gl000220'): 1, ('chr1', 'chrUn_gl000220'): 1, ('chr2', 'chrUn_gl000220'): 1, ('NONE', 'chr15'): 1, ('NONE', 'chr9'): 1, ('chr13', 'chrY'): 1, ('chrUn_gl000220', 'chr5'): 1, ('chr7_gl000195_random', 'chr7'): 1, ('chr6', 'chrUn_gl000220'): 1, ('chr5', 'chrY'): 1, ('chrM', 'chr11'): 1, ('chr14', 'chrUn_gl000211'): 1, ('NONE', 'chr11'): 1, ('chr22', 'chr13'): 1, ('chr12', 'chrUn_gl000218'): 1, ('chrUn_gl000222', 'chr7'): 1, ('chrM', 'chr6'): 1, ('chrUn_gl000220', 'chrX'): 1, ('chr8', 'chr13'): 1, ('chr21', 'chr8'): 1, ('NONE', 'chr8'): 1, ('chr20', 'chrX'): 1, ('chrM', 'chr8'): 1, ('chr1', 'chrUn_gl000221'): 1, ('chr10', 'chrUn_gl000219'): 1, ('NONE', 'chrY'): 1, ('chr19', 'chrY'): 1, ('chrUn_gl000219', 'chr9'): 1, ('chr13', 'chr22'): 1, ('chrY', 'chr21'): 1, ('chr21', 'chrUn_gl000220'): 1, ('chr21', 'chr20'): 1, ('chr21', 'chr14'): 1, ('chrUn_gl000220', 'chr17'): 1, ('chr22', 'chr5'): 1, ('chr12', 'chrY'): 1, ('chrY', 'chr8'): 1, ('chr21', 'chr7'): 1, ('chr12', 'chrUn_gl000220'): 1, ('chr15', 'chrY'): 1, ('chrY', 'chr14'): 1})
    '''
    max_k = 0
    max_success_rate = 0.0
    max_result = None
    for k in range(3, 15) :
        run = FrequencyRun(KgramSequenceClassifier(k),
                           "../data/train_sample",
                           "../data/test_sample",
                           "../data/validation_sample")
        result = run.validate()
        
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
        print "- - - - - - -"
        
    print "Best run is ", max_k, max_success_rate, max_result
        
bayes_sample_run()