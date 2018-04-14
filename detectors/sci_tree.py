#!/usr/bin/python

import numpy as np
import copy as cp
import sci_node as nd

class SciTree:
	def __init__(self, optimized=True, num_selected_attributes=2, num_trials=10):
		self._optimized = optimized
		self._num_selected_attributes = num_selected_attributes
		self._num_trials = num_trials
		self._root = None
		self._n_instances = 0
		self._n_dimensions = 0
		self._depth_limit = 0

	def build(self, data):
		self._n_instances = len(data)
		if self._n_instances <=0:
			return
		self._n_dimensions = len(data[0])-1
		self._depth_limit = self.get_random_height(self._n_instances)
		#self._root = self._recursive_build(data, 0)
		data = np.array(data)
		self._root = self._recursive_build(data, 0)

	def get_num_instances(self):
		return self._n_instances


	def _recursive_build(self, data, cur_depth=0):
		n_samples = len(data)
		if n_samples == 0:
			return None
		if n_samples <= 2 or cur_depth > self._depth_limit:
			return nd.SciNode(len(data), None, None, 0.0, 0.0, None)
			
		split_attributes, weights, split_value, boundary, left_child_data, right_child_data = self._split_data(data)
		children = {}
		children[0] = self._recursive_build(left_child_data, cur_depth+1)
		children[1] = self._recursive_build(right_child_data, cur_depth+1)
		return nd.SciNode(len(data), split_attributes, weights, split_value, boundary, children)

	
	def _split_data(self, data):
		best_split_attributes = np.random.choice(range(1, self._n_dimensions+1), self._num_selected_attributes, replace=False)
		selected_data = data[:, best_split_attributes]
		best_weights = np.array([np.random.uniform(-1.0, 1.0, self._num_selected_attributes)]).T
		best_projected_values = np.dot(selected_data, best_weights)[:, 0]
		sorted_projected_values = cp.deepcopy(best_projected_values)
		sorted_projected_values.sort()
		best_split_value = sorted_projected_values[np.random.choice(len(sorted_projected_values)-2, 1)[0]+1]

		if self._optimized:
			max_sd_gain, best_split_value = self._compute_max_sd_gain(sorted_projected_values) 
			for i in range(1, self._num_trials):
				split_attributes = np.random.choice(range(1, self._n_dimensions+1), self._num_selected_attributes, replace=False)
				selected_data = data[:, split_attributes]
				weights = np.array([np.random.uniform(-1.0, 1.0, self._num_selected_attributes)]).T
				projected_values = np.dot(selected_data, weights)[:, 0]
				sorted_projected_values = cp.deepcopy(projected_values)
				sorted_projected_values.sort()
				sd_gain, split_value = self._compute_max_sd_gain(sorted_projected_values)
				if sd_gain > max_sd_gain:
					max_sd_gain = sd_gain
					best_split_attributes = split_attributes
					best_weights = weights
					best_split_value = split_value
					best_projected_values = projected_values
	
		left_child_data = []
		right_child_data = []
		for i in range(len(best_projected_values)):
			if best_projected_values[i] <= best_split_value:
				left_child_data.append(data[i])
			else:
				right_child_data.append(data[i])
		function_values = best_projected_values - best_split_value
		boundary = function_values.max() - function_values.min()

		return best_split_attributes, best_weights, best_split_value, boundary, np.array(left_child_data), np.array(right_child_data)

	def _compute_max_sd_gain(self, sorted_values):
		value_sum = 0.0
		value_squared_sum = 0.0
		num_values = len(sorted_values)
		for v in sorted_values:
			value_sum += v
			value_squared_sum += v*v
		value_std = np.sqrt(max(0, value_squared_sum/num_values - pow(value_sum/num_values, 2.0)))
		cur_left_sum = sorted_values[0]
		squared_incremental = sorted_values[0]*sorted_values[0]
		cur_left_squared_sum = squared_incremental
		cur_right_sum = value_sum - sorted_values[0]
		cur_right_squared_sum = value_squared_sum - squared_incremental
		cur_left_std = np.sqrt(max(0, cur_left_squared_sum/1.0 - pow(cur_left_sum/1.0, 2.0)))
		cur_right_std = np.sqrt(max(0, cur_right_squared_sum/(num_values-1.0) - pow(cur_right_sum/(num_values-1.0), 2.0)))
		max_sd_gain = (value_std-(cur_left_std+cur_right_std)/2.0)/value_std	
		max_sd_gain_index = 0

		for i in range(1, num_values-1):
			cur_left_sum += sorted_values[i]
			squared_incremental = sorted_values[i]*sorted_values[i]
			cur_left_squared_sum += squared_incremental	
			cur_right_sum -= sorted_values[i]
			cur_right_squared_sum -= squared_incremental 
			cur_left_std = np.sqrt(max(0, cur_left_squared_sum/(i+1.0) - pow(cur_left_sum/(i+1.0), 2.0)))
			cur_right_std = np.sqrt(max(0, cur_right_squared_sum/(num_values-i-1.0) - pow(cur_right_sum/(num_values-i-1.0), 2.0)))
			sd_gain = (value_std-(cur_left_std+cur_right_std)/2.0)/value_std
			if sd_gain > max_sd_gain:
				max_sd_gain = sd_gain	
				max_sd_gain_index = i

		return max_sd_gain, sorted_values[max_sd_gain_index]
			

	def display(self):
		self._recursive_display(self._root)	
	
	def _recursive_display(self, sci_node, leftStr=''):
		if sci_node is None:
			return

		print leftStr+'('+str(len(leftStr))+','+str(sci_node._data_size)+'):'+str(sci_node._split_attributes)+':'+str(sci_node._weights if sci_node._weights == None else sci_node._weights.T)+':'+str(sci_node._split_value)
		
		children = sci_node.get_children()
		if not children:
			return
		for key in children.keys():
			self._recursive_display(children[key], leftStr+' ')

	def _h(self, i):
		return np.log(i)+0.5772156649

	
	def _c(self, n):
		if n > 2:
			h = self._h(n-1)	
			return 2*h -2*(n-1)/n
		if n==2:
			return 1
		else:
			return 0

	def get_norm_factor(self):
		return self._c(self._n_instances)

	def predict(self, point, depth_limit=np.inf):
		return self._recursive_get_depth(self._root, point, 0, depth_limit)

	def  _recursive_get_depth(self, sci_node, point, cur_depth=0, depth_limit=np.inf):
		if sci_node is None:
			return -1
		children = sci_node.get_children()
		if not children or cur_depth >= depth_limit:
			return cur_depth+self._c(sci_node.get_data_size())
		else:
			function_value = np.dot(np.array(point)[sci_node.get_split_attributes()], sci_node.get_weights()) - sci_node.get_split_value()
			if function_value <= 0:
				return self._recursive_get_depth(children[0], point, cur_depth+(1 if function_value >= -1.0*sci_node.get_boundary() else 0), depth_limit)
			else:
				return self._recursive_get_depth(children[1], point, cur_depth+(1 if function_value <= sci_node.get_boundary() else 0), depth_limit)
	

	def get_random_height(self, num_samples):
		return 2*np.log2(num_samples)+0.8327
