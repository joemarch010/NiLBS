
import numpy as np

from NiLBS.weighting.weighting_function import WeightingFunction


class WeightingFunctionNearestNeighbour(WeightingFunction):

    def __init__(self, points):

        self.points = points

    def evaluate(self, x, pose):

        d = np.sum((self.points - x) ** 2, axis=1)
        closest_index = np.argmin(d)
        result = np.zeros((self.points.shape[0] + 1))
        result[closest_index] = 1

        return result

    def evaluate_set(self, X, pose):

        result = np.zeros((X.shape[0], self.points.shape[0] + 1))

        for i in range(0, X.shape[0]):

            result[i] = self.evaluate(X[i], pose)

        return result