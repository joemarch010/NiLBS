
import numpy as np


class WeightingFunction:
    """


    Abstract class for determining the bone weights of a point (x, y, z)


    """

    def __init__(self, n_bones):

        self.n_bones = n_bones

    def evaluate(self, x):
        """
        :param x: Point, (x, y, z)
        :return: Array-Like, all point are assigned 1 to the root bone and 0 to all others.
        """
        result = np.zeros((self.n_bones))
        result[0] = 1

        return result