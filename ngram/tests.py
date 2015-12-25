'''
@author: bighouse
'''
import unittest

from ngram import NgramSequenceClassifier
from ngram import FrequencyModel
from interpolation import InterpolationClassifier
from laplace import AddDeltaClassifier

class Test(unittest.TestCase):

    def test_bayes_model(self):
        ls = ["AGTGCAGTT"]
        model = FrequencyModel()
        for s in ls :
            model.train(s)
        with_t = model.counters['AGT']['T'] == 1
        self.assertTrue(model.counters['AGT']['G'] == 1 and with_t)


    def test_kgram_classifier(self):
        classifier = NgramSequenceClassifier()
        classifier.train("0001", "A")
        classifier.train("0000", "B")
        classifier.train("00010000", "C")
        classifier.prepare()
        self.assertEqual('C', classifier.classify("0000000000000001"))


    def test_interpolation_1(self):
        classifier = InterpolationClassifier(2, [0.5, 0.5], verbose=False)
        classifier.train("101", "A")
        classifier.train("0000", "B")
        classifier.prepare()
        self.assertEqual('A', classifier.classify("001"))

    def test_interpolation_2(self):
        classifier = InterpolationClassifier(2, [0.5, 0.5], verbose=False)
        classifier.train("000000100000001", "A")
        classifier.train("000000000000001", "B")
        classifier.prepare()
        self.assertEqual('A', classifier.classify("001"))
        
    def test_laplace_1(self):
        classifier = AddDeltaClassifier(3, 2, verbose=True)
        classifier.train("000000100000001", "A")
        classifier.train("000000000000001", "B")
        classifier.prepare()
        self.assertEqual('A', classifier.classify("0001"))
        
    def test_laplace_2(self):
        classifier = AddDeltaClassifier(1, 2, verbose=True)
        classifier.train("01", "A")
        classifier.train("00", "B")
        classifier.prepare()
        self.assertEqual('B', classifier.classify("00"))
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()