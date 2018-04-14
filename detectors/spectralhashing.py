#! /usr/bin/python
import sys
import numpy as np
import math
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class SH:
    def __init__(self, num_bits = 8):
        self.num_bits = num_bits

    def fit(self, data):
        num_instances = len(data)
        num_features = len(data[0])
        # StandardScale.fit_transform: Center and scale the data so that the mean is 0 and std deviation is 1
        # scaled_data = StandardScaler().fit_transform(data)

        # 1) Perform PCA
        num_slice = np.min([self.num_bits, num_features])
        features = np.matrix(data).T
        cov_matrix = np.cov(features)
        eig_vals, eig_vectors = np.linalg.eig(cov_matrix)
        # Project the data on to the space defined by the eigen vectors
        if num_slice < num_features:
            selected_eig_vals = eig_vals.argsort()[-1*num_slice:][::-1]
            # print " Indexes of " + str(num_slice) + " largest eigenval"
            # print selected_eig_valsS
            eig_vectors = np.asarray(eig_vectors)[selected_eig_vals]
            # print " Corresponding eigenvectors"
            # print np.matrix(eig_vectors).T
        # print eig_vectors.T
        projected_data = np.matrix(data) * (-1 *( eig_vectors.T))

        # 2) Fit uniform distribution
        ## This will give the most outer and inner points after projections.
        ## which represents the distribution
        max_n = projected_data.max(axis=0) + sys.float_info.epsilon
        min_n = projected_data.min(axis=0) - sys.float_info.epsilon

        # 3) Enumerate eigenfunctions
        R = (max_n - min_n)
        maxMode = np.ceil((self.num_bits+1)*R/np.max(R))
        # print np.sum(maxMode)
        # print np.size(maxMode)
        nModes = np.sum(maxMode) - np.size(maxMode) +1

        modes = np.asarray(np.ones([int(nModes),num_slice]))
        m = 0
        for i in range(num_slice):
            rowStart = int(m+1)
            rowEnd = int(m + maxMode[0,i]-1)
            arr = np.reshape(range(2, int(maxMode[0,i]+1)),[-1,1])
            j =0;
            while rowStart <= rowEnd:
                modes[rowStart,i] = arr[j]
                j= j+1
                rowStart =rowStart +1
            m = m + maxMode[0,i] - 1
        # print maxMode

        modes = modes - 1
        omega0 = np.pi/R
        rmat = np.tile(omega0,(int(nModes), 1))
        # print rmat
        # omegas = modes.T.dot(np.tile(omega0,(int(nModes), 1)))
        omegas = np.multiply(modes,rmat)
        eigVal = omegas.sum(axis=1);
        eigValSort = sorted(range(len(eigVal)), key=lambda k: eigVal[k])
        eigVal = np.sort(eigVal);
        modes = modes[eigValSort[1: (self.num_bits + 1)],:]

        # Assign parameters
        self.eig_vectors = eig_vectors
        self.min_n = min_n
        self.max_n = max_n
        self.modes = modes

    # This function converts data point into binary codes
    def get_hash_value(self, data):
        num_instances = len(data)
        num_features = len(data[0])

        projections = np.matrix(data) * (-1 * self.eig_vectors.T)

        data = projections - np.tile(self.min_n,(num_instances,1))
        omega0 = np.pi/(self.max_n -self.min_n)
        omegas = np.multiply(self.modes,np.tile(omega0,(self.num_bits,1)))

        U = np.zeros([num_instances,self.num_bits])

        for i in range(self.num_bits):
            omegai = np.tile(omegas[i,:], (num_instances, 1))
            ys = np.sin(np.multiply(np.matrix(data),omegai) + np.pi/2)
            # print ys
            yi = np.prod(ys, axis=1)
            for j in range(num_instances):
                U[j,i] = yi[j]

        # Convert nums to bits 0 or 1
        binaryCodes = np.zeros([num_instances,self.num_bits])
        for i in range(num_instances):
            for j in range(self.num_bits):
                if (U[i,j] >0):
                    binaryCodes[i,j] = 1
                else:
                    binaryCodes[i,j] = 0
        return binaryCodes

    # Function to convert a single instance into binary code
    # def get_hash_value(self, data_instance):

    def get_compact_code(self, binary_codes):
        num_instances = len(binary_codes)
        num_words = math.ceil(self.num_bits/8.0)
        compact_codes = np.zeros([num_instances, int(num_words)])
        for i in range(num_instances):
            # Convert the binary code array to string and revert it
            binaryStr = ''.join(str(int(e)) for e in binary_codes[i])[::-1]
            # Devide the string into chunks of 8 bits
            binaryChunks = [binaryStr[ind:ind + 8] for ind in range(0, len(binaryStr), 8)]
            for j in range(len(binaryChunks)):
                compact_codes[i,j] = int(binaryChunks[j], 2)
        return np.flip(compact_codes, axis=1)





