#! /usr/bin/python

import numpy as np
import pandas as pd
import csv

import sci_tree as st 
import sampling as sp

class SciForest:
	def __init__(self, num_trees, sampler, optimized=True, num_selected_attributes=2, num_trials=10):
		self._num_trees = num_trees
		self._sampler = sampler
		self._optimized = optimized
		self._num_selected_attributes = num_selected_attributes
		self._num_trials = num_trials
		self._trees = []
		self._num_instances = -1
	
	def build(self, data):
		indices = range(len(data))
		# Uncomment the following code for continuous values
		data = np.c_[indices, data]
		self._num_instances = len(data)

		# Important: clean the tree array
		self._trees = []

		self._sampler.fit(data)
		sampled_datas = self._sampler.draw_samples(data)
		
		for i in range(self._num_trees):
			sampled_data = sampled_datas[i]
			tree = st.SciTree(self._optimized, self._num_selected_attributes, self._num_trials)
			tree.build(sampled_data)
			#tree.display()
			self._trees.append(tree)

	def fit(self, data):
		self.build(data)

	def display(self):
		for t in self._trees:
			t.display()

	def decision_function(self, data):
		indices = range(len(data))
		# Uncomment the following code for continuous data
		data = np.c_[indices, data]

		norm_factors = []
		for i in range(self._num_trees):
			norm_factors.append(self._trees[i].get_norm_factor())

		depths=[]
		data_size = len(data)
		for i in range(data_size):
			d_depths = []
			for j in range(self._num_trees):
				d_depths.append(self._trees[j].predict(data[i], self._trees[j].get_num_instances())*1.0/norm_factors[j])
			depths.append(d_depths)

		avg_depths=[]
		for i in range(data_size):
			depth_avg = 0.0
			for j in range(self._num_trees):
				depth_avg += depths[i][j] 
			depth_avg /= self._num_trees
			avg_depths.append(depth_avg)

		avg_depths = np.array(avg_depths)
		return -1.0*2**(avg_depths)

#num_trees = 1
#bootstrap = False 
#forest = SciForest(num_trees, sp.VSSampling(num_trees), True, 2, 10)
#
##Stackloss dataset; dimension is 4
##data = pd.read_csv('dat/stackloss.csv')
##X = data.as_matrix()[:, 1:].tolist()
#
### Covertype dataset; dimension is 10
#data = pd.read_csv('../dat/covertype.csv', header=None)
#X = data.as_matrix()[:, :-1].tolist()
#
## Anomaly dataset; dimension is 2
##data = pd.read_csv('dat/anomaly.csv', header=None)
##X = data.as_matrix()[:, :-1].tolist()
#
#forest.fit(X)
#forest.display()
#print 'scores: \n', forest.decision_function(X)
