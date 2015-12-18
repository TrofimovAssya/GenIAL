import numpy as np

from convnet import ConvolutionalNetwork
from sklearn.cross_validation import train_test_split 
from sklearn.metrics import classification_report

# Dataset
X = np.loadtxt('../feature/3grams_count_mc_features')
y = np.loadtxt('../data/tag_mc')
X -= X.min()
X /= X.max()
X = X.reshape(-1, 1, 8, 8)
y = np.asarray(y)
y = y.astype(np.int32)
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Instanciation
convnet = ConvolutionalNetwork()

# Train
convnet.train(X_train, y_train)

# Report
preds = convnet.predict(X_test)
tags = y_test 
print classification_report(tags, preds)