import matplotlib.pyplot as plt
import numpy as np
import json
import math
import pickle

from masking import create_time_step_mask
from utils import euclidean_distance, distance_matrix_from_projection


class DistanceMatrix:

    def __init__(self):
        self.member_names = None
        self.num_time_steps = None
        self.matrix = None
        self.raw_data = None
        self.time_points = None

    def copy(self):
        copy = DistanceMatrix()
        copy.member_names = self.member_names.copy()
        copy.num_time_steps = self.num_time_steps.copy()
        copy.matrix = self.matrix.copy()
        if self.raw_data is not None:
            copy.raw_data = self.raw_data.copy()
        if self.time_points is not None:
            copy.time_points = self.time_points.copy()
        return copy

    def get_num_dimensions(self):
        return len(self.matrix)

    def get_reference_idx(self, reference_member_idx):
        offsets = [0]
        for i, time_steps in enumerate(self.num_time_steps):
            offsets.append(offsets[i] + int(time_steps))

        return offsets[reference_member_idx], offsets[reference_member_idx + 1]

    def get_time_step_indices(self, time_idx, reference_idx=None):
        indices = []
        offset = 0
        for i, time_steps in enumerate(self.num_time_steps):
            if i == reference_idx:
                continue
            if time_idx < time_steps:
                indices.append(offset + time_idx)
            offset += time_steps

        return indices

    def crop_to_num_time_steps(self, min_num_time_steps):
        print('[DistanceMatrix] Cropping distance matrix to number of time steps')

        num_members = len(self.member_names)
        num_time_steps = self.num_time_steps

        if min_num_time_steps > min(num_time_steps):
            min_num_time_steps = min(num_time_steps)

        n = num_members * min_num_time_steps
        d = np.zeros((n, n))

        last_offset_i = 0
        for i, offset_i in enumerate(num_time_steps):
            last_offset_j = 0
            for j, offset_j in enumerate(num_time_steps):
                for k in range(min_num_time_steps):
                    for l in range(min_num_time_steps):
                        d[i * min_num_time_steps + k][j * min_num_time_steps + l] = \
                            self.matrix[last_offset_i + k][last_offset_j + l]
                last_offset_j += offset_j
            last_offset_i += offset_i

        self.matrix = d
        self.num_time_steps = [min_num_time_steps for _ in range(num_members)]

    def crop_to_min_num_time_steps(self):
        self.crop_to_num_time_steps(min(self.num_time_steps))

    def extend_to_max_num_time_steps(self):
        print('[DistanceMatrix] Extending distance matrix to max. number of time steps')

        num_members = len(self.member_names)
        num_time_steps = self.num_time_steps
        max_num_time_steps = max(num_time_steps)

        n = num_members * max_num_time_steps
        d = np.zeros((n, n))

        last_offset_i = 0
        for i, offset_i in enumerate(num_time_steps):
            last_offset_j = 0
            for j, offset_j in enumerate(num_time_steps):
                for k in range(offset_i):
                    for l in range(offset_j):
                        d[i * max_num_time_steps + k][j * max_num_time_steps + l] = \
                            self.matrix[last_offset_i + k][last_offset_j + l]
                    for l in range(offset_j, max_num_time_steps):
                        d[i * max_num_time_steps + k][j * max_num_time_steps + l] = \
                            self.matrix[last_offset_i + k][last_offset_j + offset_j - 1]
                for k in range(offset_i, max_num_time_steps):
                    for l in range(offset_j):
                        d[i * max_num_time_steps + k][j * max_num_time_steps + l] = \
                            self.matrix[last_offset_i + offset_i - 1][last_offset_j + l]
                    for l in range(offset_j, max_num_time_steps):
                        d[i * max_num_time_steps + k][j * max_num_time_steps + l] = \
                            self.matrix[last_offset_i + offset_i - 1][last_offset_j + offset_j - 1]
                last_offset_j += offset_j
            last_offset_i += offset_i

        self.matrix = d
        self.num_time_steps = [max_num_time_steps for _ in range(num_members)]

    def resample_time(self, resampled_number_of_timesteps, same_length=True):

        if self.time_points is None:
            print('No time points set')
            return self

        positions = self.raw_data
        time_points = self.time_points

        if positions is None:
            from techniques.classicalmds import ClassicalMdsEmbedding
            positions = ClassicalMdsEmbedding(self).project()

        common_time_interval_start = max([min(time_points[i]) for i in range(len(time_points))])
        common_time_interval_end = min([max(time_points[i]) for i in range(len(time_points))])
        if same_length:
            common_time_frames = np.linspace(common_time_interval_start, common_time_interval_end, num=resampled_number_of_timesteps)
        else:
            max_interval_end = max([max(time_points[i]) for i in range(len(time_points))])
            common_time_frames = np.linspace(common_time_interval_start, max_interval_end, num=resampled_number_of_timesteps)

        resampled_data_points = [] #np.zeros((len(self.member_names)*resampled_number_of_timesteps, positions.shape[0]))

        from scipy.interpolate import interp1d
        last_offset = 0
        for i, offset in enumerate(self.num_time_steps):

            run_positions = positions[:, last_offset:last_offset + offset].T
            run_time_points = time_points[i]

            interpolator = interp1d(run_time_points, run_positions, axis=0, kind='linear', bounds_error=False, fill_value=(np.nan, np.nan))
            interpolated_data = interpolator(common_time_frames)

            nan_positions = np.where(np.isnan(interpolated_data))
            if len(nan_positions[0]) > 0:
                num_time_steps = nan_positions[0][0]
            else:
                num_time_steps = None
            resampled_data_points.append(interpolated_data[:num_time_steps])

            self.time_points[i] = common_time_frames[:num_time_steps].tolist()
            last_offset += offset

        self.num_time_steps = [len(resampled_data_points[i]) for i in range(len(resampled_data_points))]
        stacked_data = np.vstack(resampled_data_points)
        self.raw_data = stacked_data.T
        self.matrix = distance_matrix_from_projection(stacked_data)

        return self


    def to_file(self, filename):
        print('[DistanceMatrix] Saving distance matrix to file: {}'.format(filename))

        data = {
            'matrix': pickle.dumps(self.matrix).decode('latin1'),
            'num_time_steps': self.num_time_steps,
            'member_names': self.member_names,
            'raw_data': pickle.dumps(self.raw_data).decode('latin1'),
            'time_points': self.time_points
        }

        json_data = json.dumps(data)
        with open(filename, "w") as file:
            file.write(json_data)

    def from_file(self, filename):
        print('[DistanceMatrix] Loading distance matrix from file: {}'.format(filename))

        with open(filename, "r") as file:
            data = json.load(file)

        self.matrix = pickle.loads(data['matrix'].encode('latin1'))
        self.raw_data = pickle.loads(data['raw_data'].encode('latin1'))

        self.time_points = data['time_points'] if 'time_points' in data else None
        self.num_time_steps = data['num_time_steps']
        self.member_names = data['member_names']

        return self


def create_untangle_distance_matrix(distance_matrix, reference, alpha=0.5):
    """
    Note: This function is deprecated!

    This function updates the distance matrix such that the distance between
    the specified reference and the target positions is minimized.
    """
    distance_matrix = distance_matrix.copy()
    matrix = distance_matrix.matrix

    # Gather distances between nodes.
    distances = [0]
    for i in range(reference[0] + 1, reference[1]):
        distances.append(distances[-1] + matrix[i - 1, i])

    # Update distance matrix.
    for i in range(reference[0], reference[1]):
        for j in range(i, reference[1]):
            current_distance = matrix[i, j]
            target_distance = distances[j-reference[0]] - distances[i-reference[0]]
            matrix[i, j] = matrix[j, i] = max((1-alpha), 0) * current_distance + alpha * target_distance

    return distance_matrix


def create_modified_distance_matrix(distance_matrix, reference_idx, target_distances):
    target_distance_matrix = distance_matrix.copy()
    target_distance_matrix.matrix[reference_idx[0]:reference_idx[1], reference_idx[0]:reference_idx[1]] = target_distances
    return target_distance_matrix


def create_extended_distance_matrix(distance_matrix, reference_idx, target_distances):
    """
    This function adds a constraining / target run to the specified distance matrix
    and set respective distances, i.e.:
    - distances between reference members and target positions shall be zero
    - distances between target position and other members shall be equal to the distances between
     the reference run and the other members.

    The weights of the 'artificial' distances shall be set to the specified factor.
    """
    n_ref = reference_idx[1] - reference_idx[0]
    n_dim = distance_matrix.get_num_dimensions()
    n = n_dim + n_ref

    d = np.zeros((n, n))

    # First copy the part of the distance matrix that does not change.
    d[:n_dim, :n_dim] = distance_matrix.matrix

    # Now copy the distances from reference run to other runs.
    d[:n_dim, n_dim:] = distance_matrix.matrix[:, reference_idx[0]:reference_idx[1]]
    d[n_dim:, :n_dim] = distance_matrix.matrix[reference_idx[0]:reference_idx[1], :]

    # Set distances from reference run to the target.
    #target_distances_ = distance_matrix.matrix[reference[0]:reference[1], reference[0]:reference[1]].copy()
    d[n_dim:, reference_idx[0]:reference_idx[1]] = target_distances
    d[reference_idx[0]:reference_idx[1], n_dim:] = target_distances.T

    # Set the target distances.
    d[n_dim:, n_dim:] = target_distances

    constrained = distance_matrix.copy()
    constrained.matrix = d
    constrained.member_names.append('target')
    constrained.num_time_steps.append(n_ref)

    return constrained


def create_1D_time_distance_matrix(distance_matrix):
    """
    This function updates the distance matrix such that the distance between
    the specified reference and the target positions is minimized.
    """

    #mask = create_time_step_mask(distance_matrix)

    distance_matrix = distance_matrix.copy()
    matrix = distance_matrix.matrix

    num_time_steps = distance_matrix.num_time_steps

    last_offset_i = 0
    for i, offset_i in enumerate(num_time_steps):
        last_offset_j = 0
        for j, offset_j in enumerate(num_time_steps):
            for k in range(offset_i):
                for l in range(offset_j):
#                    if mask[last_offset_i + k][last_offset_j + l] == 0:
                    temporal_dist = abs(k - l)
                    matrix[last_offset_i + k][last_offset_j + l] = temporal_dist
            last_offset_j += offset_j
        last_offset_i += offset_i

    return distance_matrix


def test_first_idea():

    def dummy():
        return np.array([(0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
                         (0, 0), (2, 1), (0, 2), (-2, 3), (0, 4)])

    matrix = DistanceMatrix()
    matrix.num_time_steps = [5, 5]
    matrix.member_names = ['const', 'sine']
    reference_run = 1

    #matrix.matrix = np.zeros((10, 10))
    matrix.matrix = distance_matrix_from_projection(dummy())
    print(matrix.matrix)
    print('-----------')

    # Generate 2D embedding from which we derive a lower dimensional distance matrix.
    from techniques.ours import OurEmbedding
    emb = OurEmbedding(matrix, matrix, 2, None, np.ones_like(matrix.matrix), reference_run, dummy().T, draw_iterations=False)
    matrix.matrix = distance_matrix_from_projection(emb.project().T)

    mask = (create_time_step_mask(matrix) == 0)
    target = matrix.copy()
    new_distances = create_1D_time_distance_matrix(matrix)
    target.matrix[mask] = new_distances.matrix[mask]
    print(target.matrix)

    weights = np.ones_like(matrix.matrix)
    weights[mask] = 0.1
    print(weights)
    emb = OurEmbedding(matrix, target, 2, None, weights, reference_run, dummy().T, draw_iterations=False)

    import matplotlib.pyplot as plt
    from plotting import plot, plot_heatmap
    #plot_heatmap(matrix.matrix)
    plot(emb.project(), matrix, reference_run)


def test_square_example():

    num_points = 4

    points = []
    for i in range(num_points):
        for j in range(num_points):
            points.append((i, j))

    n = len(points)
    matrix = np.zeros((n, n))
    weights = np.ones_like(matrix)
    for i in range(n):
        for j in range(n):
            matrix[i, j] = euclidean_distance(points[i], points[j])

    matrix[0, 1] = matrix[1, 0] = 0.0
    weights[0, 1] = weights[1, 0] = 100

    modified_distances = np.array([
        [0, 2, 0, 0],
        [2, 0, 0, 0],
        [0, 0, 0, 2],
        [0, 0, 2, 0]
    ])

    a = 1
#    weights = np.array([
#        [1, a, 1, 1],
#        [a, 1, 1, 1],
#        [1, 1, 1, a],
#        [1, 1, a, 1]
#    ])
#    weights = None

    from techniques.ours import smacof
    res, _, _ = smacof(matrix, None, weights, None, 100, eps=0)

    fig, ax = plt.subplots()
    for i in range(n):
        for j in range(i, n):
            ax.plot([res[i][0], res[j][0]], [res[i][1], res[j][1]])
        circle = plt.Circle(res[i], 0.1, fill=True, color='blue')
        ax.add_patch(circle)
        ax.text(res[i][0], res[i][1], str(i), ha='center', va='center', color='white')
    plt.show()



if __name__ == '__main__':

    test_first_idea()
    # test_square_example()