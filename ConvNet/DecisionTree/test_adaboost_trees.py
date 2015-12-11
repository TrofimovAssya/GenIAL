from sklearn.cross_validation import train_test_split 
from sklearn import metrics
import numpy as np
from adaboost_trees import AdaboostTrees

# load data
X = np.loadtxt('../feature/5grams_count_mc_features')
y = np.loadtxt('../data/tag_mc')
X -= X.min()
X /= X.max()
X_train, X_test, y_train, y_test = train_test_split(X, y)

#instanciate forest
forest = AdaboostTrees(20)
forest.train(X_train, y_train)
expected = y_test
predicted = forest.predict(X_test)

# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))