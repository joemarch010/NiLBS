
import numpy as np

from weighting.weighting_function import WeightingFunction


class WeightingFunctionPointwise(WeightingFunction):
    """


    Weighting function which is defined point-wise at the vertices as the artist's defined vertex weights
    and (1, 0, ..., 0) everywhere else.


    """

    def __init__(self, vertices, weights):
        """
        :param vertices: Numpy array-like, Vx3
        :param weights: Numpy array-like, VxB
        """
        self.point_map = dict()
        self.n_weights = weights.shape[1]

        for i in range(0, vertices.shape[0]):

            self.point_map[vertices[i].data.tobytes()] = weights[i]

    def evaluate(self, x):

        if x.data.tobytes() in self.point_map:

            return self.point_map[x.data.tobytes()]

        result = np.zeros((self.n_weights))
        result[0] = 1

        return result