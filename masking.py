import numpy as np


def create_time_step_mask(distance_matrix):
    """
    This function returns a mask such that every entry of the
    specified distance matrix is set to 1 if it corresponds to a
    pair of equal time steps, 0 otherwise.
    """
    mask = np.zeros(distance_matrix.matrix.shape)
    num_time_steps = distance_matrix.num_time_steps

    last_offset_i = 0
    for i, offset_i in enumerate(num_time_steps):
        last_offset_j = 0
        for j, offset_j in enumerate(num_time_steps):
            min_num_time_steps = min(offset_i, offset_j)
            for k in range(min_num_time_steps):
                mask[last_offset_i + k][last_offset_j + k] = 1

            last_offset_j += offset_j
        last_offset_i += offset_i

    return mask


def create_single_time_step_mask(distance_matrix, t):
    """
    This function returns a mask such that every entry of the
    specified distance matrix is set to 1 if it corresponds to the
    specified time step.
    """
    mask = np.zeros(distance_matrix.matrix.shape)
    num_time_steps = distance_matrix.num_time_steps

    last_offset_i = 0
    for i, offset_i in enumerate(num_time_steps):
        last_offset_j = 0
        for j, offset_j in enumerate(num_time_steps):
            mask[last_offset_i + t][last_offset_j + t] = 1
            last_offset_j += offset_j
        last_offset_i += offset_i

    return mask


def create_reference_only_mask(distance_matrix, reference):
    """
    This functions creates an index mask that only contains distances
    that both involve the reference.
    """
    mask = np.zeros(distance_matrix.matrix.shape)
    mask[reference[0]:reference[1], reference[0]:reference[1]] = 1
    return mask


def restrict_mask_to_reference(mask, reference):
    """
    This function takes an index mask and restricts it further to a reference,
    i.e., entries not corresponding to the specified interval will be set to 0.
    Additionally, the entries corresponding to only the reference will be set to 0.
    """
    # Upper left.
    mask[:reference[0], :reference[0]] = 0
    # Upper right.
    mask[reference[1]:, :reference[0]] = 0
    # Lower left.
    mask[:reference[0], reference[1]:] = 0
    # Lower right.
    mask[reference[1]:, reference[1]:] = 0
    # Square.
    mask[reference[0]:reference[1], reference[0]:reference[1]] = 0
    return mask

