from matplotlib import pyplot as plt

from masking import create_single_time_step_mask
from techniques.embedding import Embedding
from utils import *
from techniques.classicalmds import ClassicalMdsEmbedding

import numpy as np
from scipy.spatial import procrustes


def create_distance_matrix_for_time_step(distance_matrix, t):
    """
    This function returns the sub-matrix of distance_matrix that contains only the values for the
    specified time step.
    """
    mask = create_single_time_step_mask(distance_matrix, t)
    sub = distance_matrix.matrix[mask != 0]

    n = len(distance_matrix.num_time_steps)
    dm = distance_matrix.copy()
    dm.matrix = sub.reshape((n, n))

    return dm


class ProcrustesEmbedding(Embedding):
    def __init__(self, distance_matrix, num_dimensions, mask, weights, reference_run, draw_iterations=False):
        self.distance_matrix = distance_matrix
        self.num_dimensions = num_dimensions
        self.mask = mask
        self.weights = weights
        self.reference_run = reference_run
        self.reference_idx = distance_matrix.get_reference_idx(reference_run)
        self.draw_iterations = draw_iterations

    def project(self):

        num_time_steps = self.distance_matrix.num_time_steps

        proj = np.zeros((sum(num_time_steps), self.num_dimensions))

        reference_projection = None

        def plot_state(distance_matrix, p):
            if not self.draw_iterations:
                return
            plt.clf()
            for i in range(len(distance_matrix.num_time_steps)):
                plt.scatter(p.T[0], p.T[1], label=distance_matrix.member_names[i])
            plt.legend()
            plt.draw_if_interactive()
            plt.pause(0.01)

        def do_step(t):
            dm = create_distance_matrix_for_time_step(self.distance_matrix, t)
            # X, _, _ = smacof(dm.matrix, None, reference_projection)
            X = ClassicalMdsEmbedding(dm, 2).project().T

            plot_state(dm, X)

            if reference_projection is not None:
                # FIXME: when using smacof, estimateAffinePartial2D works. When using Classical MDS it doesn't.
                _, X, _ = procrustes(reference_projection, X)
                # M, _ = cv.estimateAffinePartial2D(reference_projection, X)
                # X = cv.transform(X.reshape(-1, 1, 2), M)

            # Subtract the reference run's position.
            X -= X[self.reference_run]

            return X

        # First time step is reference for the procrustes analysis.
        reference_projection = do_step(0)
        offset = 0
        for i in range(len(num_time_steps)):
            proj[offset] = reference_projection[i]
            offset += num_time_steps[i]

        for t in range(1, num_time_steps[self.reference_run]):

            X = do_step(t)

            # Set the respective entries and add the time step offset.
            offset = 0
            for i in range(len(num_time_steps)):
                proj[offset+t] = X[i] + [t, 0]
                offset += num_time_steps[i]

        return proj.T
