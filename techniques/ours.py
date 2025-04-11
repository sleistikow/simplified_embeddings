import math

from masking import create_reference_only_mask, create_time_step_mask
from techniques.embedding import Embedding
from utils import *

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

from weighting import create_weights_for_modified_distance_matrix, create_time_step_weights, \
    create_weights_for_extended_distance_matrix


def smacof(dissimilarities, initial_projection=None, weights=None, mask=None, max_iter=100, eps=1e-5, callback=None):
    """
    SMACOF (Scaling by MAjorizing a COmplicated Function) algorithm.
    Adapted from sklearn.
    """
    n_samples = dissimilarities.shape[0]

    if initial_projection is None:
        initial_projection = np.random.RandomState(seed=42).randn(n_samples, 2)
    else:
        initial_projection = initial_projection.copy()

    u = 1.0
    if weights is not None:
        V = -weights + np.diag(weights.sum(axis=1))
        try:
            u = np.linalg.pinv(V)
        except np.linalg.LinAlgError:
            print('SVD did not converge, returning initial projection')
            return initial_projection, 0.0, 0
    else:
        u /= n_samples

    X = initial_projection

    disparities = dissimilarities.copy()
    if weights is not None:
        disparities *= weights

    if mask is not None:
        disparities[mask == 0] = 0

#    if callback:
#        distances = distance_matrix_from_projection(X)
#        stress = stress_ex(distances, dissimilarities, weights=weights)
#        callback(X, disparities, stress, 0)

    old_stress = None
    for it in range(max_iter):
        distances = distance_matrix_from_projection(X)

        # Compute stress.
        stress = stress_ex(distances, dissimilarities, weights=weights)

        # Apply mask.
        if mask is not None:
            distances[mask == 0] = 1e-5

        # Update X using the Guttman transform.
        distances[distances == 0] = 1e-5
        ratio = disparities / distances
        B = -ratio
        B[np.arange(len(B)), np.arange(len(B))] += ratio.sum(axis=1)

        # Do the update.
        X = np.dot(u, np.dot(B, X))

        if callback:
            callback(X, distances, stress, it + 1)

        if old_stress is not None:
            if old_stress - stress < eps:
                break
        old_stress = stress

    return X, stress, it + 1


class OurEmbedding(Embedding):
    def __init__(self, original_distance_matrix, target_distance_matrix, num_dimensions, mask, weights, reference_run, initial_projection, num_iterations=100, draw_iterations=True, store_iterations=False, ax=None, stress_tol=None, decay=None):
        self.original_distance_matrix = original_distance_matrix
        self.target_distance_matrix = target_distance_matrix
        self.num_dimensions = num_dimensions
        self.mask = mask
        self.weights = weights
        self.reference_run = reference_run
        self.reference_idx = original_distance_matrix.get_reference_idx(reference_run)
        self.initial_projection = initial_projection.copy() if initial_projection is not None else None
        self.num_iterations = num_iterations
        self.draw_iterations = draw_iterations
        self.ax = ax
        self.store_iterations = store_iterations
        self.iterations = []
        self.omega = None
        self.stress_tol = stress_tol
        self.decay = decay

        # Unused, see 'project_stepwise':
        self.target_positions = generate_target_positions(self.original_distance_matrix.matrix, self.reference_idx, self.num_dimensions)

    def pull_to_target(self, projection, factor):
        # Comment: Unused code, see 'project_stepwise'.
        if self.target_positions is None:
            return
        p = projection[self.reference_idx[0]:self.reference_idx[1], :]
        vectors = self.target_positions - p
        p += vectors * factor
        return np.linalg.norm(vectors)

    def project_stepwise(self, plot_state):
        # Comment: I leave this (unused) code here on purpose.
        # We used it to see what happens if after an iteration of smacof, all points
        # are pulled back to some fixed position (e.g., the target positions).
        # However, this is equal to another random initialization to SMACOF and
        # hence was discarded.

        threshold = 1e-6
        factor = 0.5

        positions = self.initial_projection.T

        # (Optionally) set the reference run to the target positions.
        # positions[self.reference_idx[0]:self.reference_idx[1], :] = self.target_positions

#        error = 0
        for i in range(self.num_iterations):

            print('pulling')
            for _ in range(1):
                error = self.pull_to_target(positions, factor)

            print('optimizing')
            for _ in range(1):
                new_positions, _, _ = smacof(self.target_distance_matrix.matrix, positions, self.weights, self.mask, 1, callback=plot_state)

                delta = np.linalg.norm(new_positions - positions)
                if delta < threshold:
                    break

                positions = new_positions

        plt.clf()

        return positions

    def project(self):

        positions = self.initial_projection.T

        weighted_stress_history = []
        reference_stress_history = []
        classical_stress_history = []
        self.iterations = []

        from plotting import plot
#        plot(positions.T, self.distance_matrix, interactive=False)

        reference_mask = create_reference_only_mask(self.original_distance_matrix, self.reference_idx)

        mask = None
        # HACK: If the original distance matrix is equal to the target distance matrix,
        # this means that we optimizing an extended version of the original matrix.
        # If they differ, the original one has been modified (i.e. our old approach).
        # In both cases, the original stress (i.e., the deviation from the original matrix)
        # is calculated using only the entries that can also be found in the original matrix.
        # For the extended case we hence need to mask those entries.
        if self.original_distance_matrix == self.target_distance_matrix:
            n = self.original_distance_matrix.get_num_dimensions()
            n = n - (self.reference_idx[1] - self.reference_idx[0])
            mask = np.zeros_like(self.target_distance_matrix.matrix)
            mask[:n, :n] = 1  # Ignore the extended entries.

        # This is simply a plotting callback that is called each iteration.
        def plot_state(X, distances, stress, it):
            if not self.draw_iterations:
                return

            # Calculate classical stress.
            classical_stress = stress_ex(distances, self.original_distance_matrix.matrix, mask=mask)

            # Calculate reference stress.
            reference_stress = stress_ex(distances, self.target_distance_matrix.matrix, mask=reference_mask)

            # Log values.
            weighted_stress_history.append(stress)
            reference_stress_history.append(reference_stress)
            classical_stress_history.append(classical_stress)

            if self.store_iterations:
                it = (X.T.copy(), stress, reference_stress, classical_stress)
                self.iterations.append(it)

            # Plot state.
            # title = f'Stress: {stress:.2f}, Iteration: {it}'
            title = f'Stress: {stress:.4f}, Target: {reference_stress:.4f}'
            if self.ax is None:
                plt.clf()
            else:
                self.ax.clear()
            plot(X.T, self.original_distance_matrix, self.reference_run, title=title, ax=self.ax, interactive=True, equal_axis=False)

            plt.pause(0.01)

        if self.weights is not None:
            # In case we have no weights, we can simply use the original SMACOF algorithm.
            new_positions, _, _ = smacof(self.target_distance_matrix.matrix, positions, self.weights, self.mask, self.num_iterations, callback=plot_state)
            #new_positions = self.project_stepwise(plot_state)
        else:
            # Otherwise, we apply weighted MDS.

            def weighted_mds(omega, max_num_iterations, plotting):
                plotting_callback = plot_state if plotting else None
                if mask is None: # If we not mask, we create weights for the original distance matrix.
                    weights = create_weights_for_modified_distance_matrix(self.original_distance_matrix, self.reference_idx, omega)
                else: # If we have a mask, that means we are optimizing an extended distance matrix.
                    weights = create_weights_for_extended_distance_matrix(self.original_distance_matrix, self.reference_idx, omega)
                if self.decay is not None: # Optionally, apply a temporal decay.
                    weights *= create_time_step_weights(self.original_distance_matrix, self.decay)
                return smacof(self.target_distance_matrix.matrix, positions, weights, self.mask, max_num_iterations, callback=plotting_callback)

            original_projection = distance_matrix_from_projection(self.initial_projection.T)
            original_stress = stress_ex(original_projection, self.original_distance_matrix.matrix, mask=mask)
            original_stress_class = get_kruskal_stress_class(original_stress)
            #_, unweighted_stress, _ = weighted_mds(1, self.num_iterations, False)

            # Plot initial result.
            plot_state(self.initial_projection.T, original_projection, original_stress, 0)

            # We have to choose some bounds for omega.
            # One can simply increase the upper bound, but it will increase the time needed for the binary search.
            bounds = (1, 1000)

            if self.stress_tol is None:

                # If the stress tolerance is None, it means we want to find some optimal value for omega,
                # such that stress is not too high compared to the original stress.
                # This definition is, however, far from clear.
                # Hence this branch is very much untested. It should not be used, until
                # the methodology is clear. (In our application it is not used.)

                print('WARNING: This branch is untested and should not be used.')

                def total_stress(omega):
                    positions, _, _ = weighted_mds(omega, 20, False)
                    distances = distance_matrix_from_projection(positions)

                    # TODO: which stress function do we have to look at here?
                    stress = stress_ex(distances, self.original_distance_matrix.matrix)
                    this_stress_class = get_kruskal_stress_class(stress)
                    class_difference = original_stress_class - this_stress_class
                    penalty = min(0, class_difference)
                    return stress - 1000 * penalty

                    # stress = stress_ex(distances, self.target_distance_matrix.matrix)
                    # reference_stress = stress_ex(distances, self.target_distance_matrix.matrix, mask=reference_mask)
                    # return stress + reference_stress

                res = minimize_scalar(total_stress, method='bounded', bounds=bounds, tol=0.1)
                omega = res.x
                print(f'Found omega={omega} in {res.nit} iterations')

            else:

                # In case we do have a stress tolerance, we try to find a value for omega
                # such that the stress does not exceed the tolerance.

                max_omega_distances = distance_matrix_from_projection(weighted_mds(bounds[1], 100, False)[0])
                max_omega_stress = stress_ex(max_omega_distances, self.original_distance_matrix.matrix, mask=mask)

                max_stress = original_stress + self.stress_tol
                if max_omega_stress < max_stress:
                    print('tolerance will not be reached')
                    # max_stress = min(max_omega_stress, max_stress)

                def idx_to_omega(idx):
                    return idx+1

                def binary_search(arr, target):
                    low = 0
                    high = len(arr) - 1
                    result = -1

                    while low <= high:
                        mid = (low + high) // 2

                        if arr[mid] < target:
                            result = mid
                            low = mid + 1
                        else:
                            high = mid - 1

                    return result

                class StressFunction:
                    def __init__(self, original_distance_matrix):
                        self.original_distance_matrix = original_distance_matrix

                    def __getitem__(self, idx):
                        omega = idx_to_omega(idx)
                        X, weighted_stress, _ = weighted_mds(omega, 50, False)
                        #return weighted_stress
                        distances = distance_matrix_from_projection(X)
                        stress = stress_ex(distances, self.original_distance_matrix.matrix, mask=mask)
                        return stress

                    def __len__(self):
                        return bounds[1] - bounds[0] + 1

                idx = binary_search(StressFunction(self.original_distance_matrix), max_stress)
                if idx == -1:
                    idx = 0
                omega = idx_to_omega(idx)

            print(f'Omega: {omega}')
            self.omega = omega # Store final omega.

            # Apply the omega.
            new_positions, _, _ = weighted_mds(omega, self.num_iterations, True)

        return new_positions.T


