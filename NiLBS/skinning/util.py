
import sys

import numpy as np


def redistribute_weights(weights, bone_hierarchy, active_bones):
    """

    Redistributes vertex bone weights from an B-dimensional vector to an B'-dimensional vector, redistributing unused
    bone weights to the parent bone weights.

    :param weights: Numpy array-like, VxB, initial vertex weights.
    :param bone_hierarchy: Numpy array-like, 2xB, bone parent map.
    :param active_bones: List, B', list of bone indices whose weights should be preserved.
    :return: Numpy array-like, VxB', redistributed bone weights.
    """
    result = np.zeros((weights.shape[0], len(active_bones)))

    for i in range(0, weights.shape[0]):

        initial_weights = weights[i]

        for j in range(0, initial_weights.shape[0]):

            k = j
            while k not in active_bones and k != -1:
                k = bone_hierarchy[0][k]

            if k == -1:
                # Root bone not included in active bones? This is a problem
                sys.exit("Error: When redistributing weights, the root bone must be active.")

            result[i][k] += initial_weights[j]

    return result
