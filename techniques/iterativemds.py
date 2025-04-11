from techniques.embedding import Embedding
from sklearn.manifold import MDS


class IterativeMdsEmbedding(Embedding):

    def __init__(self, distance_matrix, num_dimensions):
        self.distance_matrix = distance_matrix.matrix
        self.num_dimensions = num_dimensions

    def project(self):
        return MDS(n_components=self.num_dimensions, dissimilarity='precomputed', random_state=0).fit_transform(self.distance_matrix).T
