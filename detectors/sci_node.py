#! /usr/bin/python

class SciNode:
	def __init__(self, data_size, split_attributes=None, weights=None, split_value=0.0, boundary = 0.0, children=None):
		self._data_size = data_size
		self._children = children
		self._split_attributes = split_attributes
		self._weights = weights
		self._split_value = split_value
		self._boundary = boundary

	def display(self):
		print self._split_attributes
		print self._weights
		print self._split_value
		print self._boundary
		print self._data_size
		print self._children
	
	def get_children(self):
		return self._children

	def get_data_size(self):
		return self._data_size

	def get_split_attributes(self):
		return self._split_attributes

	def get_weights(self):
		return self._weights
	
	def get_split_value(self):
		return self._split_value

	def get_boundary(self):
		return self._boundary

