#!/usr/bin/python
# Spectral hashing implementation

import numpy as np
import pandas as pd
from detectors import SH

# 1) Prepare the data
# ---------------------
# A ) load data from csv
# Get features and class label
input_data = pd.read_csv('datasets/test.csv', header=None)
X = input_data.as_matrix()[:, :-1].tolist()
ground_truth = input_data.as_matrix()[:, -1].tolist()

num_instances = len(X)
num_features = len(X[0])

# 2) Train spectral hashing
# -------------------------
# Using a different number of bits for encoding
code_len = [2, 4, 8, 16, 32, 64]

for num_bits in code_len:
    # Initialize classifier
    print("Spectral Hashing:")
    classifier = SH(num_bits)
    # Train classifer
    classifier.fit(X)
    # Compress data
    # classifier.get_binary_code(X)
    # for x in x:
        # classifier.get_binary_code(x)






