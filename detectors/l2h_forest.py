#! /usr/bin/python
import numpy as np
import copy as cp

import sh_tree as shtree

class L2HForest:

    def __init__(self, num_trees, algo):
        self._num_trees = num_trees
        self._algo = algo
        self._trees = []

    def fit(self, data):
        self.build(data)

    def build(self, data):
        self._algo.fit(data)
        self._binaryCodes = self._algo.get_hash_value(data)
        self._compactCodes = self._algo.get_compact_code(self._binaryCodes)

        for i in range(self._num_trees):
            tree = shtree.L2HTree(self._algo)
            tree.build(self._binaryCodes)
            self._trees.append(tree)

    def decision_function(self, data):
        depths = []
        data_size = len(data)
        for i in range(data_size):
            d_depths = []
            for j in range(self._num_trees):
                transformed_data = data[i]
                d_depths.append(self._trees[j].predict( transformed_data))
            depths.append(d_depths)

        # Arithmatic mean
        avg_depths = []
        for i in range(data_size):
            depth_avg = 0.0
            for j in range(self._num_trees):
                depth_avg += depths[i][j]
            depth_avg /= self._num_trees
            avg_depths.append(depth_avg)

        avg_depths = np.array(avg_depths)
        # return -1.0*avg_depths
        return avg_depths

    def get_avg_branch_factor(self):
        sum = 0.0
        for t in self._trees:
            sum += t.get_avg_branch_factor()
        return sum / self._num_trees

    def get_binary_codes(self):
        return self._binaryCodes