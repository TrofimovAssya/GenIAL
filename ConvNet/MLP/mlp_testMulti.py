import numpy as np
from sklearn.cross_validation import train_test_split 
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer
from mlp import MultilayerPerceptron


#data fetching

X = np.loadtxt('../data/readsMulti')
y = np.loadtxt('../data/chrsMulti')


#data normalization

X -= X.min()
X /= X.max()


#instantiation

mlp = MultilayerPerceptron([100,300,160,60,4])


#data splitting

X_train, X_test, y_train, y_test = train_test_split(X, y)
labels_train = LabelBinarizer().fit_transform(y_train)
labels_test = LabelBinarizer().fit_transform(y_test)


#information prints

print 'Total Train Examples: ' + str(X_train.shape[0])
print 'Total Test Examples: ' + str(X_test.shape[0])
print 'Total Examples: ' + str(X_test.shape[0] + X_train.shape[0])


#training

mlp.train(X_train,labels_train,epochs=30000)


#predictions

predictions = []
for i in range(X_test.shape[0]):
    output = mlp.prediction(X_test[i])
    predictions.append(np.argmax(output))


#result prints

print confusion_matrix(y_test,predictions)
print classification_report(y_test,predictions)
