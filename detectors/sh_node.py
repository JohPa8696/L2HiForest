class SHNode:
    def __init__(self, data_size=0, children={}, children_count={}):
        # self._data = data
        self._data_size = data_size
        self._children = children
        self._children_count = children_count

    def display(self):
        # print self._data
        print self._data_size
        print self._children
        print self._children_count

    def get_children(self):
        return self._children

    def get_data_size(self):
        return self._data_size

    def get_children_count(self):
        return self._children_count
