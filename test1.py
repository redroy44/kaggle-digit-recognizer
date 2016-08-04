import sys

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn import cross_validation as cv
from sklearn.ensemble import RandomForestClassifier

dataset = pd.read_csv("data/train.csv")

print dataset.shape

train, test = cv.train_test_split(dataset, train_size = 0.8)

train_Y = train.iloc[:, 0]
train_X = train.iloc[:, 1:]

val_Y = test.iloc[:, 0]
val_X = test.iloc[:, 1:]

print train_X.shape, train_Y.shape
print val_X.shape, val_Y.shape

rf = RandomForestClassifier(n_estimators=100, n_jobs=8)
rf.fit(train_X, train_Y)
pred = rf.predict(val_X)

result =  float(sum(pred == val_Y)) / val_Y.shape[0]

print "%.2f%%" %(result * 100)

scores = cv.cross_val_score(rf, train_X, train_Y, cv=5)

print scores

#np.savetxt('submission_rand_forest.csv', np.c_[range(1,len(test)+1),pred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')







