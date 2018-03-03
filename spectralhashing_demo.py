#!/usr/bin/python
# Spectral hashing implementation

import numpy as np
import pandas as pd
from detectors import SH

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

for num_bits in code_len:
    # Initialize classifier
    print("Spectral Hashing:")
    print "Number of Bits" + str(num_bits)
    classifier = SH(num_bits)
    # Train classifer
    classifier.fit(X)
    # Compress data
    binaryCodes = classifier.get_hash_value(X)
    compactCode = classifier.get_compact_code(binaryCodes)
    for r in compactCode:
        print r






