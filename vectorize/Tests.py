'''
Created on Sep 29, 2015

@author: bighouse
'''
import unittest
from vectorize.Vectorizer import Vectorizer

class Test(unittest.TestCase):

    def test_convert_letter(self):
        vectorizer = Vectorizer()
        print vectorizer
        self.assertEqual([0, 1, 0, 0], list(vectorizer.encode_letter("G")))
        self.assertEqual([1, 0, 0, 0], list(vectorizer.encode_letter("T")))
        self.assertEqual([0, 0, 1, 0], list(vectorizer.encode_letter("C")))
        self.assertEqual([1, 0, 0, 0], list(vectorizer.encode_letter("A")))


if __name__ == "__main__":
    unittest.main()