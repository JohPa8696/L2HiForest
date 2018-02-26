#! /usr/bin/python
import sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class SH:
    def __init__(self, num_bits = 8):
        self.num_bits = num_bits

    def fit(self, data):
        num_instances = len(data)
        num_features = len(data[0])
        # StandardScale.fit_transform: Center and scale the data so that the mean is 0 and std deviation is 1
        scaled_data = StandardScaler().fit_transform(data)

        # 1) Perform PCA
        num_slice = np.min([self.num_bits , num_features])
        features = np.matrix(scaled_data).T
        cov_matrix = np.cov(features)
        eig_vals, eig_vectors = np.linalg.eig(cov_matrix)
        # Project the data on to the space defined by the eigen vectors
        print "Eigenvalues"
        print eig_vals
        print "Eigenvectors"
        print eig_vectors
        if num_slice < num_features:
            selected_eig_vals = eig_vals.argsort()[-1*num_slice:][::-1]
            print " Indexes of " + str(num_slice) + " largest eigenval"
            print selected_eig_vals
            eig_vectors = np.asarray(eig_vectors)[selected_eig_vals]
            print " Corresponding eigenvectors"
            print np.matrix(eig_vectors).T
        projected_data = np.matrix(data).dot(np.matrix(eig_vectors).T)
        # projected_data = np.matrix(data).dot(eig_vectors)

        # 2) Fit uniform distribution
        max_n = projected_data.max(axis=0) - sys.float_info.epsilon
        min_n = projected_data.min(axis=0) - sys.float_info.epsilon

        # 3) Enumerate eigenfunctions
        R = (max_n - min_n)
        maxMode = np.ceil((self.num_bits+1)*R/np.max(R))
        print np.sum(maxMode)
        print np.size(maxMode)
        nModes = np.sum(maxMode) - np.size(maxMode) + 1

        modes = np.asarray(np.ones([int(nModes),num_slice]))
        m = 0
        for i in range(num_slice):
            modes[int(m+1):int(m + maxMode[0,i]-1), int(i)] = range(2, int(maxMode[0,i]))
            m = m + maxMode[0,i] - 1
        print maxMode

        modes = modes - 1
        omega0 = np.pi/R
        rmat = np.tile(omega0,(int(nModes), 1))
        print rmat
        omegas = modes.dot(np.tile(omega0,(int(nModes), 1)).T)
        eigVal = -1 * np.sum(omegas^ 2, 2);
        [yy, ii] = np.sort(eigVal);
        modes = modes[ii[2: self.num_bits + 1],:]

        # Assign parameters
        self.eigenvectors = eig_vectors
        self.min_n = min_n
        self.max_n = max_n
        self.modes = modes

    # This function converts data point into binary codes
    def get_binary_code(self, data):
        num_instances = len(data)
        num_features = len(data[0])

        projections = np.matrix(data).dot(np.matrix(self.eigenvectors).T)

        datax = projections - np.tile(self.min_n,(num_instances,1))
        omega0 = np.pi/(self.max_n -self.min_n)
        omegas = self.modes.dot(np.tile(omega0,(self.num_bits,1)))

        U = np.zeros([num_instances,self.num_bits])

        for i in range(self.num_bits):
            omegai = np.tile(omegas[i,:], (num_instances, 1))
            ys = np.sin(np.matrix(data).dot(omegai) + np.pi/2)
            yi


