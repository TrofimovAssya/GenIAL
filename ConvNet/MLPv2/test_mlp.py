import numpy as np

from sklearn.cross_validation import train_test_split 
from sklearn.metrics import classification_report
from multilayer_perceptron  import MultilayerPerceptronClassifier

# Dataset
X = np.loadtxt('../feature/5grams_count_mc_features')
y = np.loadtxt('../data/tag_mc')
X -= X.min()
X /= X.max()
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Instanciation
mlp = MultilayerPerceptronClassifier(activation='relu', hidden_layer_sizes = (20,), max_iter = 200)

# Train
mlp.fit(X_train, y_train)

# Report
preds = mlp.predict(X_test)
tags = y_test 
print classification_report(tags, preds)