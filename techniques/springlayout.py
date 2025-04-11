from techniques.embedding import Embedding
from utils import generate_target_positions

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class SpringLayoutEmbedding(Embedding):

    def __init__(self, distance_matrix, num_dimensions, mask, initial_projection=None, reference_run=None, num_iterations=1000, draw_iterations=True):
        self.distance_matrix = distance_matrix
        self.num_dimensions = num_dimensions
        self.mask = mask
        self.initial_projection = initial_projection
        self.reference_run = reference_run
        self.refence_idx = distance_matrix.get_reference_idx(self.reference_run)
        self.num_iterations = num_iterations
        self.draw_iterations = draw_iterations
        self.target_positions = generate_target_positions(self.distance_matrix.matrix, self.refence_idx)

    def project(self):

        initial_pos = None
        if self.initial_projection is not None:
            pos = self.initial_projection.T
            initial_pos = {i: pos[i] for i in range(len(pos))}

        n = self.distance_matrix.matrix.shape[0]
        similarities = 1 - self.distance_matrix.matrix

        G = nx.Graph()

        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if self.mask is None or self.mask[i][j] > 0:
                    edges.append((i, j, similarities[i][j]))
        G.add_weighted_edges_from(edges)

        k = 0.1
        threshold = 1e-4

        if not self.draw_iterations:
            pos = nx.spring_layout(G, pos=initial_pos, fixed=None, seed=0, iterations=self.num_iterations, weight='weight', k=k)
        else:
            pos = initial_pos
            steps = 10
            num_iter = int(self.num_iterations / steps)
            for _ in range(num_iter):
                new_pos = nx.spring_layout(G, pos=pos, fixed=None, seed=0, iterations=steps, weight='weight', k=k, threshold=threshold)

                p = self.pos_to_array(pos)

                from plotting import plot
                plt.clf()
                plot(p, self.distance_matrix, self.reference_run, interactive=True)
                plt.pause(0.01)

                if np.linalg.norm(self.pos_to_array(new_pos) - p) < threshold:
                    break

                pos = new_pos

            plt.clf()

        return self.pos_to_array(pos)

    def pos_to_array(self, pos):
        as_list = [p for p in pos.values()]
        return np.asarray(as_list).reshape((self.num_dimensions, -1))
