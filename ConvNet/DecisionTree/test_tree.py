from sklearn.cross_validation import train_test_split 
from sklearn import metrics
import numpy as np
from tree import DecisionTree

# load data

X = np.loadtxt('../feature/5grams_count_mc_features')
y = np.loadtxt('../data/tag_mc')
X -= X.min()
X /= X.max()
X_train, X_test, y_train, y_test = train_test_split(X, y)

tree = DecisionTree()
tree.train(X_train, y_train)
expected = y_test
predicted = tree.predict(X_test)

# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))