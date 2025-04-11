from techniques.embedding import Embedding

import numpy as np


class ClassicalMdsEmbedding(Embedding):

    def __init__(self, distance_matrix, num_dimensions=None):
        self.distance_matrix = distance_matrix.matrix
        self.num_dimensions = num_dimensions
        self.eigenvalues = None

    def project(self):
        (n, n) = self.distance_matrix.shape
        E = (-0.5 * self.distance_matrix ** 2)

        Er = np.asmatrix(np.mean(E, 1))
        Es = np.asmatrix(np.mean(E, 0))

        F = np.array(E - np.transpose(Er) - Es + np.mean(E))

        [U, S, V] = np.linalg.svd(F.astype(np.float64), full_matrices=True, hermitian=True)

        Y = U * np.sqrt(S)

        projected_points = Y[:, 0:self.num_dimensions]
        self.eigenvalues = S[0:self.num_dimensions]

        return projected_points.T

    def get_eigenvalues(self):
        return self.eigenvalues
