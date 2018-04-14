#! /usr/bin/python

import numpy as np
import pandas as pd
import csv
import time
import matplotlib.pyplot as mpl
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import IsolationForest

from detectors import VSSampling
from detectors import SciForest

rng = np.random.RandomState(42)
num_ensemblers = 100

# data = pd.read_csv('dat/glass.csv', header=None)
data = pd.read_csv('datasets/glass.csv', header=None) #data is a dataframe object
X = data.as_matrix()[:, :-1].tolist()
ground_truth = data.as_matrix()[:, -1].tolist()

classifiers = [("SciForest", SciForest(num_ensemblers, VSSampling(num_ensemblers),4))]
results = []

for i, (clf_name, clf) in enumerate(classifiers):
	print "	"+clf_name+":"
	# prediction stage
	start_time = time.time()
	clf.fit(X)
	train_time = time.time()- start_time
	# evaluation stage
	y_pred = clf.decision_function(X).ravel()
	# y_pred.sort()
	# print y_pred
	test_time = time.time()- start_time - train_time
	auc = roc_auc_score(ground_truth, y_pred)
	results.append(-1.0*auc*100)
	print "AUC:	", auc
	# print "Training time:	", train_time
	# print "Testing time:	", test_time
