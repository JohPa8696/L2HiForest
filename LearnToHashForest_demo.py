#!/usr/bin/python
# Spectral hashing implementation

import numpy as np
from sklearn.metrics import roc_auc_score
import pandas as pd
from detectors import SH
from detectors import L2HTree
from detectors import L2HForest
# 1) Prepare the data
# ---------------------
# A ) load data from csv
# Get features and class label
input_data = pd.read_csv('datasets/glass.csv', header=None)
X = input_data.as_matrix()[:, :-1].tolist()
ground_truth = input_data.as_matrix()[:, -1].tolist()
# X = pd.read_csv('datasets/testdata.csv', header=None).as_matrix().tolist()
# X = np.array(X)
# X = X * [1,0.5]
num_instances = len(X)
num_features = len(X[0])

# 2) Train spectral hashing
# -------------------------
# Using a different number of bits for encoding
code_len = [ 16, 32, 64]

num_tree = 100
classifiers = [("Spectral Hashing", L2HForest(num_tree, SH(16)))]
for i, (clf_name, clf) in enumerate(classifiers):
    # Initialize classifier
    print "	" + clf_name + ":"
    # print "Number of Bits" + str(num_bits)
    # Train classifer
    clf.fit(X)
    binaryCodes = clf.get_binary_codes()
    # Pred
    y_pred = clf.decision_function(binaryCodes).ravel()
    # Compress data
    auc = roc_auc_score(ground_truth, y_pred)
    print "AUC:	", auc






