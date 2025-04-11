from techniques.embedding import Embedding
from scipy.optimize import minimize, shgo, newton
import math

from utils import *


class OptimizerEmbedding(Embedding):

    def __init__(self, distance_matrix, num_dimensions, mask, reference_run, initial_projection, max_iter=100):

        # The shape of x0 determines also the shape (and order) of the "working variables".
        self.x0 = self.to_linear(initial_projection)

        # The original distance matrix.
        self.distance_matrix = distance_matrix

        # The number of dimensions to project to.
        self.num_dimensions = num_dimensions

        self.reference_run = reference_run
        self.max_iter = max_iter

        self.mask = mask
        self.p = self.distance_matrix.matrix[self.mask != 0].flatten()

        self.constraints = self.generate_constraints(reference_run, num_dimensions)
        self.bounds = self.generate_bounds()

    def to_linear(self, projection):
        return projection.flatten()

    def from_linear(self, x):
        return np.asarray(x).reshape(self.num_dimensions, int(len(x)/self.num_dimensions))

    def stress_function(self, x):

        # Calculate projected distances.
        projection = self.from_linear(x).T
        distances = distance_matrix_from_projection(projection)

        d = distances[self.mask != 0].flatten()
        p = self.p  # precalculated

        # Calculate stress.
        stress = 0
        for i in range(len(d)):
            stress += (d[i] - p[i])**2

        # We intentionally ignore dividing by the constant here.

        # Taking the square root might not be necessary...
        stress = math.sqrt(stress)

        # TODO: Add stress from reference run.

        return stress

    def generate_constraints(self, reference_run, axis=1):
        constraints = []

        x_offset = 0
        for i in range(reference_run[0], reference_run[1]):
            index = i * self.num_dimensions
            # TODO: these do not work with SLSQP solver.
#            constraints.append({'type': 'eq',
#                                'fun': lambda x: x[index] - x_offset})
#            constraints.append({'type': 'eq',
#                                'fun': lambda x: x[index + axis]})
            x_offset += 1

        return constraints

    def generate_bounds(self):
        min_val = min(self.x0)
        max_val = max(self.x0)
        val_range = max_val - min_val
        min_val -= val_range * 0.5
        max_val += val_range * 0.5

        return [(min_val, max_val) for _ in range(len(self.x0))]

    def project(self):

        from plotting import plot
        import matplotlib.pyplot as plt

        def plot_state(x):
            plot(self.from_linear(x).T, self.distance_matrix, interactive=True)
            plt.pause(0.01)

        # sol = shgo(self.stress_function, self.bounds, constraints=self.constraints, iters=1)
        # sol = minimize(self.stress_function, self.x0, method='L-BFGS-B', bounds=self.bounds, constraints=self.constraints, callback=plot_state)
        sol = minimize(self.stress_function, self.x0, method='SLSQP', bounds=self.bounds, constraints=self.constraints, callback=plot_state)
        if not sol.success:
            print(f'[OptimizerEmbedding] Could not solve: {sol.message}')

        plt.clf()

        return self.from_linear(sol.x)