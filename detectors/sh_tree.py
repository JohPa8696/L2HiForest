import numpy as np

import sh_node as shnode

class LTHTree:

    def __init__(self, algo):
        self.algo = algo

    def build(self, data):
        self.num_samples = len(data)
        self.root = self._recursive_build(data)

    def _recursive_build(self,data, bit_index = 0):
        num_samples = len(data)
        if num_samples == 0:
            return None
        if num_samples == 1:
            return shnode.SHNode(num_samples,{},{})
        else:
            num_bits = len(data[0])
            # Select a random number of bits to split, this is to introduce variations to the trees
            # And subsequently, increase the robustness
            rand_bits = np.random.choice([1,2])
            cur_index = bit_index
            partition = self._split_data(data,rand_bits, cur_index)
            while len(partition) == 1 and cur_index < num_bits:



    def _split_data(self, data, numbits):

