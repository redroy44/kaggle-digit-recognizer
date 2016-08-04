import sys

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn import cross_validation

dataset = pd.read_csv("data/train.csv")

#print dataset.head()

labels = dataset.iloc[:, 0]
train = dataset.iloc[:, 1:]

print labels.shape

print train.shape







