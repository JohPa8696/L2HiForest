import numpy as np

import sh_node as shnode

class L2HTree:

    def __init__(self, algo, rand_branching_factor = [1,2] ):
        self.algo = algo
        self._rand_branching_factor = rand_branching_factor

    def build(self, data):
        self._num_samples = len(data)
        self._root = self._recursive_build(data)
        self._branch_factor = self._get_avg_branch_factor()
        self._reference_path_length = self.get_random_path_length_symmetric(self._num_samples)

    def _recursive_build(self, data, bit_index = 0):
        num_samples = len(data)
        if num_samples == 0:
            return None
        if num_samples == 1:
            return shnode.SHNode(num_samples,{},{}, bit_index)
        else:
            num_bits = len(data[0])
            if bit_index >= num_bits:
                return None
            # Select a random number of bits to split, this is to introduce variations to the trees
            # And subsequently, increase the robustness
            rand_bits = np.random.choice(self._rand_branching_factor)
            cur_index = bit_index
            partition = self._split_data(data, rand_bits, cur_index)
            cur_index += rand_bits
            while len(partition) == 1 and cur_index < num_bits:
                rand_bits = np.random.choice(self._rand_branching_factor)
                partition = self._split_data(data,rand_bits,cur_index)
                cur_index += rand_bits
            if cur_index >= num_bits:
               return shnode.SHNode(len(data),{}, {}, cur_index - rand_bits)
            children_count = {}
            for key in partition.keys():
                children_count[key] = len(partition.get(key))

            mean = np.mean(children_count.values())
            std = np.std(children_count.values())

            children = {}
            for key in partition.keys():
                child_data = partition.get(key)
                children[key] = self._recursive_build(child_data, cur_index)
            return shnode.SHNode(len(data), children, children_count, cur_index - rand_bits)

    def _split_data(self, data, numbits, cur_index):
        ''' Split the data using LSH '''
        partition = {}
        for i in range(len(data)):
            key = data[i][cur_index:(cur_index+numbits)]
            key = ''.join(str(int(e)) for e in key)
            if key not in partition:
                partition[key] = [data[i]]
            else:
                sub_data = partition[key]
                sub_data.append(data[i])
                partition[key] = sub_data
        return partition

    # Display the tree
    def display(self):
        self._recursive_display(self._root)

    def _recursive_display(self, node, leftStr=''):
        if node is None:
            return
        children = node.get_children()

        print leftStr + '(Node_ind:' + str(len(leftStr)) + ', split_ind:' + str(node._index) + ')- Num_child:' + str(
            node._data_size) + ', Child_counts:' + str(node._children_count)

        for key in children.keys():
            self._recursive_display(children[key], leftStr + ' ')

    # Get the path length of the data point
    def predict(self, data_point):
        # point = self._lsh.format_for_lsh(np.mat(point)).A1
        path_length = self._recursive_get_search_depth(self._root,0, data_point, 0)
        return pow(2.0, (-1.0 * path_length / self._reference_path_length))

    # Get the depth of the data point
    def _recursive_get_search_depth(self, node, cur_depth, data_point, cur_index, granularity=1):
        if node is None:
            return -1
        children = node.get_children()
        if not children:
            real_depth = node.get_index()
            adjust_factor = self.get_random_path_length_symmetric(node.get_data_size())
            return cur_depth * np.power(1.0 * real_depth / max(cur_depth, 1.0), granularity) + adjust_factor
        else:
            # key = self._lsh.get_hash_value(data_point[1:], node.get_index())
            bit_chunk = len(children.keys()[0])
            key = data_point[cur_index: (cur_index + bit_chunk)]
            key = ''.join(str(int(e)) for e in key)
            if key in children.keys():
                return self._recursive_get_search_depth(children[key], cur_depth + 1, data_point, cur_index + bit_chunk)
            else:
                cur_depth = cur_depth + 1
                real_depth = node.get_index() + 1
                return cur_depth * np.power(1.0 * real_depth / max(cur_depth, 1), granularity)

    def _recursive_sum_BF(self, lsh_node):
        if lsh_node is None:
            return None, None
        children = lsh_node.get_children()
        if not children:
            return 0, 0
        else:
            i_count, bf_count = 1, len(children)
            for key in children.keys():
                i_c, bf_c = self._recursive_sum_BF(children[key])
                i_count += i_c
                bf_count += bf_c
            return i_count, bf_count

    def _get_avg_branch_factor(self):
        i_count, bf_count = self._recursive_sum_BF(self._root)
        # Single node PATRICIA trie
        if i_count == 0:
            return 2.0
        return bf_count * 1.0 / i_count

    def get_random_path_length_symmetric(self, num_samples):
        if num_samples <= 1:
            return 0
        elif num_samples > 1 and num_samples <= round(self._branch_factor):
            return 1
        else:
            return (np.log(num_samples) + np.log(self._branch_factor - 1.0) + 0.5772) / np.log(
                self._branch_factor) - 0.5