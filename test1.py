import sys

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn import cross_validation as cv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

dataset = pd.read_csv("data/train.csv")
data_test = pd.read_csv("data/test.csv")

print dataset.shape
print data_test.shape

train, test = cv.train_test_split(dataset, train_size = 0.8)

train_Y = train.iloc[:, 0]
train_X = train.iloc[:, 1:]

val_Y = test.iloc[:, 0]
val_X = test.iloc[:, 1:]

#test_Y = data_test.iloc[:, 0]
#test_X = data_test.iloc[:, 1:]

print train_X.shape, train_Y.shape
print val_X.shape, val_Y.shape

rf = RandomForestClassifier(n_estimators=400, n_jobs=8)
rf.fit(train_X, train_Y)
pred = rf.predict(val_X)

result =  float(sum(pred == val_Y)) / val_Y.shape[0]

print "%.2f%%" %(result * 100)

scores = cv.cross_val_score(rf, train_X, train_Y, cv=5)

#sklearn.learning_curve.learning_curve
# GridSearchCV

print np.mean(scores)

#print confusion_matrix(pred, val_Y)

pred_final = rf.predict(data_test)
np.savetxt('submission_rand_forest.csv', np.c_[range(1,data_test.shape[0]+1),pred_final], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')

print "Script finished"







