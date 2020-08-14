
import numpy as np

class VoxelMatrix:
    """

    Utility class for representing a voxel matrix which can be sub-divided ad scaled


    """
    def __init__(self, resolution):
        """
        :param resolution: [Int], (n, m, p) size of the voxel grid
        """

        self.resolution = resolution
        self.matrix = np.full((resolution[0], resolution[1], resolution[2]), False)
        self.backing_scale = [1, 1, 1]

    def set(self, point):
        """
        :param point: the point to set
        :return:
        """
