from main import Processor

from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout

class NGramProcessor(Processor):
    
    def model(self):
        model = Sequential()
        model.add(Embedding(max_features, 256))
        model.add(LSTM(256, 128, activation='sigmoid', inner_activation='hard_sigmoid'))
        model.add(Dropout(0.5))
        model.add(Dense(128, 1))
        model.add(Activation('sigmoid'))
    
    def __init__(self):
        self.samfile = None
        
    
    def register(self, samfile):
        self.samfile = samfile
    
    def process(self, x):
        rname = self.samfile.getrname(x.tid)
        seq = x.seq
    
    def report(self):
        model.compile(loss='binary_crossentropy', optimizer='rmsprop')
        model.fit(X_train, Y_train, batch_size=16, nb_epoch=10)
        score = model.evaluate(X_test, Y_test, batch_size=16)
    
    
    
