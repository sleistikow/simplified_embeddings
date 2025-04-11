import numpy as np
from utils import *


def transform_1st_pc_over_time(pc, distance_matrix, reference_run):
    """
    This function takes a projection vector and projects the first Principal Component over time.
    If a reference is provided, the reference will be subtracted.
    """

    # Pick first principal component.
    if len(pc.shape) > 1:
        pc = pc[0]

    num_time_steps = distance_matrix.num_time_steps

    y_ref = None
    if reference_run is not None:
        reference_idx = distance_matrix.get_reference_idx(reference_run)
        reference_data = pc[reference_idx[0]:reference_idx[1]]
        y_ref = np.ones((max(num_time_steps))) * reference_data[-1]
        y_ref[:len(reference_data)] = reference_data

    result = [[], []]

    last_offset = 0
    for i, offset in enumerate(num_time_steps):
        y = pc[last_offset:(last_offset + offset)]
        if y_ref is not None:
            y = (y - y_ref[:len(y)])

        if distance_matrix.time_points is None:
            x = range(len(y))
        else:
            x = distance_matrix.time_points[i]

        result[0].extend(x)
        result[1].extend(y)

        last_offset += offset

    return np.asarray(result)


def transform_n_pcs_over_time(pc, distance_matrix, reference_run):
    """
    This function takes a projection vector and projects the distance to the reference (must not be None)
    over time, while only considering the provided principal components.
    """

    reference_idx = distance_matrix.get_reference_idx(reference_run)
    num_time_steps = distance_matrix.num_time_steps
    min_num_time_steps = min(num_time_steps)

    result = [[], []]

    last_offset = 0
    for i, offset in enumerate(num_time_steps):
        y = pc[:, last_offset:(last_offset + offset)]
        y_ref = pc[:, reference_idx[0]:reference_idx[1]]

        y = np.subtract(y[:, :min_num_time_steps], y_ref[:, :min_num_time_steps])
        y = np.square(y)
        y = np.sum(y, axis=0)
        y = np.sqrt(y)

        if distance_matrix.time_points is None:
            x = range(len(y))
        else:
            x = distance_matrix.time_points[i]

        result[0].extend(x)
        result[1].extend(y)

        last_offset += offset

    return np.asarray(result)


def transform_distance_over_time(distance_matrix, reference_run):
    """
    This function takes a projection vector and projects the high dimensional distances (i.e., not the
    embedded distances) to the reference (must not be None) over time.
    """

    matrix = distance_matrix.matrix
    reference_idx = distance_matrix.get_reference_idx(reference_run)
    num_time_steps = distance_matrix.num_time_steps
    min_num_time_steps = min(num_time_steps)
    num_members = len(num_time_steps)

    result = [[], []]

    for i in range(num_members):
        x = []
        y = []
        member_idx = distance_matrix.get_reference_idx(i)
        for j in range(min_num_time_steps):
            if distance_matrix.time_points is None:
                x.append(j)
            else:
                x.append(distance_matrix.time_points[i][j])
            y.append(matrix[reference_idx[0]+j][member_idx[0]+j])

        result[0].extend(x)
        result[1].extend(y)

    return np.asarray(result)
