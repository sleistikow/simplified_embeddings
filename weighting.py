import numpy as np


def create_time_step_weights(distance_matrix, decay=lambda x: 1-x, k_size=None):
    """
    This function returns weights such that every entry of the
    specified distance matrix is set to 1 if it corresponds to a
    pair of equal time steps, and a decay is applied otherwise.
    """
    if decay is None:
        return np.ones(distance_matrix.matrix.shape)

    weights = np.zeros(distance_matrix.matrix.shape)
    num_time_steps = distance_matrix.num_time_steps

    # If not kernel size is provided, we extend it to the maximum possible.
    # It will anyway be applied only once.
    if k_size is None:
        k_size = max(num_time_steps)
        if k_size % 2 == 0:
            k_size += 1

    kernel = np.zeros((k_size, k_size))
    for i in range(k_size):
        for j in range(k_size):
            distance = (abs(i-k_size//2) + abs(j-k_size//2))
            kernel[i][j] = decay(distance)

    # kernel /= np.max(kernel) # Makes no difference!

    def apply_kernel(i, j):

        hk = k_size // 2

        i_start = max(0, i - hk)
        i_end   = min(weights.shape[0], i + hk + 1)
        j_start = max(0, j - hk)
        j_end   = min(weights.shape[1], j + hk + 1)

        subarray = weights[i_start:i_end, j_start:j_end]
        kernel_subset = kernel[i_start-(i-hk):k_size-((i+hk+1)-i_end), j_start-(j-hk):k_size-((j+hk+1)-j_end)]

        weights[i_start:i_end, j_start:j_end] = np.maximum(subarray, kernel_subset)

    last_offset_i = 0
    for i, offset_i in enumerate(num_time_steps):
        last_offset_j = 0
        for j, offset_j in enumerate(num_time_steps):
            min_num_time_steps = min(offset_i, offset_j)
            for k in range(min_num_time_steps):
                apply_kernel(last_offset_i + k, last_offset_j + k)

            last_offset_j += offset_j
        last_offset_i += offset_i

    return weights


def create_weights_for_modified_distance_matrix(distance_matrix, reference, omega):
    n = distance_matrix.get_num_dimensions()
    weights = np.ones((n, n))

    weights[reference[0]:reference[1], reference[0]:reference[1]] = omega

    return weights


def create_weights_for_extended_distance_matrix(constrained_distance_matrix, reference, omega):

    n_ref = reference[1] - reference[0]
    n = constrained_distance_matrix.get_num_dimensions()
    n_dim = n - n_ref

    weights = np.ones((n, n))

    # First set the weights for the original distances.
    # weights[:n_dim, :n_dim] = 1

    # Now set the weights from reference run to other runs.
    # weights[:n_dim, n_dim:] = omega
    # weights[n_dim:, :n_dim] = omega

    # Set the weights for distances from reference run to the target.
    # weights[n_dim:, reference[0]:reference[1]] = omega
    # weights[reference[0]:reference[1], n_dim:] = omega

    # Set the target distances.
    weights[n_dim:, n_dim:] = omega

    return weights


def wip_experiment_with_weights(distance_matrix, reference_idx, adjusted_weights):
    """
    This function is a work in progress.
    I have tested some hypothesis here regarding the weights of the extended distance matrix.
    Feel free to test your own.
    """

    adjusted_weights = adjusted_weights.copy()

    n_ref = reference_idx[1] - reference_idx[0]
    n_dim = distance_matrix.get_num_dimensions()

    # adjusted_weights[n_dim:, :n_dim] = 0
    # adjusted_weights[:n_dim, n_dim:] = 0

    # adjusted_weights[n_dim:, reference_idx[0]:reference_idx[1]] = np.identity(n_ref) * np.max(adjusted_weights)
    # adjusted_weights[reference_idx[0]:reference_idx[1], n_dim:] = np.identity(n_ref) * np.max(adjusted_weights)

    # TODO: the following line weigh the lower right corner without decay!
    # It showed some improvement over the previous version.
    adjusted_weights[n_dim:, n_dim:] = 1

    return adjusted_weights