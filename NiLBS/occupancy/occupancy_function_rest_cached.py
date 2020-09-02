
import numpy as np


from NiLBS.skinning.skinning_mesh_lbs import LBSDeformer
from NiLBS.occupancy.occupancy_function import OccupancyFunction


class OccupancyFunctionRestCached(OccupancyFunction):
    """

    Occupancy function backed by an OccupancyFunction at rest, a Pose, and a WeightingFunction, as described in the LiLBS
    technical report.


    """
    def __init__(self, rest_occupancy_function, pose, weighting_function):

        self.rest_occupancy_function = rest_occupancy_function
        self.pose = pose
        self.weighting_function = weighting_function
        self.lbs_deformer = LBSDeformer(weighting_function)

    def evaluate(self, x):

        weights = self.weighting_function.evaluate(x, self.pose)
        x_reproj = self.lbs_deformer.invert_lbs(np.array([x]), self.pose)

        return (1 - weights[weights.shape[0] - 1]) * self.rest_occupancy_function.evaluate(x_reproj)

    def evaluate_set(self, X):
        """
        Evaluate the function at a set of points
        :param X: {(x, y, z)} (Numpy array-like)
        :return: {o | o in [0, 1]}
        """

        weight_set = self.weighting_function.evaluate_set(X)
        X_reproj = self.lbs_deformer.invert_lbs((X, self.pose))
        E = self.rest_occupancy_function.evaluate_set(X_reproj)

        return (1 - weight_set[:][weight_set.shape[1] - 1]) * E

