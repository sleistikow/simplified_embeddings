import numpy as np
import math

from masking import restrict_mask_to_reference


def stress(projected_distances, original_distances, indices=None, weights=None, normalize=True):
    """
    This function calculates the stress between d and p.
    Optionally, an index array can be specified with only the indices
    set to 1 that shall be considered for stress calculation.
    """
    if indices is not None:
        projected_distances = projected_distances[np.where(indices != 0)]
        original_distances = original_distances[np.where(indices != 0)]

    difference = np.subtract(projected_distances, original_distances)
    squared_difference = np.square(difference)
    if weights is not None:
        if indices is not None:
            weights = weights[np.where(indices != 0)]
        squared_difference = np.multiply(squared_difference, weights)

    nominator = np.sum(squared_difference)

    if normalize:
        denominator = np.sum(np.square(original_distances))

        if np.all(denominator == 0):
            return 0

        return math.sqrt(np.divide(nominator, denominator))

    return nominator


def stress_ex(projected_distances, original_distances, reference=None, mask=None, weights=None, normalize=True):

    if reference is not None:
        if mask is None:
            mask = np.ones(original_distances.shape)
        restrict_mask_to_reference(mask, reference)

    return stress(projected_distances, original_distances, mask, weights, normalize)

