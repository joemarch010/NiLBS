
import numpy as np


class WeightingFunction:
    """


    Abstract class for determining the bone weights of a point (x, y, z)


    """

    def __init__(self, n_bones):

        self.n_bones = n_bones

    def generate_query(self, x, pose):

        return None

    def generate_query_set(self, X, pose):

        return None

    def evaluate(self, x, pose):
        """
        :param x: Point, (x, y, z)
        :return: Array-Like, all point are assigned 1 to the root bone and 0 to all others.
        """
        result = np.zeros((self.n_bones))
        result[0] = 1

        return result

    def evaluate_set(self, X, pose):
        """

        Evaluate the function at a set of points.

        :param X: Numpy array-like, N x 3, points to evaluate.
        :param pose: Pose, pose to query
        :return: Numoy array-like, NxB, result of the weighting function at the points.
        """

        result = np.zeros(X.shape[0], self.n_bones)

        return result