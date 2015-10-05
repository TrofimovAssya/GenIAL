'''
    Tool for data exploration. Get a feel of your data.
@author: bighouse
'''
class StatsCollector(object):

    def __init__(self, example_limit = 10):
        self.example_limit = example_limit
        self.stats = {}
        self.examples = {}
        self.global_max = None
        self.global_min = None
        self.total = 0
    
    def add(self, value, *labels):
        self.total += 1
        
        if self.global_max is None :
            self.global_max = (labels, value)
        elif self.global_max[1] < value :
            self.global_max = (labels, value)
            
        if self.global_min is None :
            self.global_min = (labels, value)
        elif self.global_min[1] > value :
            self.global_min = (labels, value)        
        for label in labels :
            self.__add_for_label(value, label)
            
    def sample(self, example, *labels):
        for label in labels :
            self.__sample_for_label(example, label)
    
    def __sample_for_label(self, example, label):
        if label in self.examples :
            ls = self.examples
            if len(ls) < self.example_limit :
                ls.append(example)
        else :
            self.examples[label] = [example]
    
    def __add_for_label(self, value, label):
        
        if label not in self.stats :
            self.stats[label] = (value, 1, value, value)
        else :
            e = self.stats[label]
            #Making some sense out of a tuple
            avg = e[0]
            count = e[1]
            mini = e[2]
            maxi = e[3]
            
            new_min = value if value < mini else mini
            new_max = value if value > maxi else maxi
            
            self.stats[label] = ((avg * count + value + 0.0) / (count + 1),
                               count + 1,
                               new_min,
                               new_max
                               )
        