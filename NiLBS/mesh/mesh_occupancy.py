
import numpy as np

from trimesh.voxel.ops import points_to_marching_cubes
from trimesh.voxel.ops import points_to_indices

class MeshOccupancy:
    """


    Mesh which is backed by an occupancy function.


    """
    def __init__(self, occupancy_function, iso_level, bounds, resolution = [64, 64, 64]):

        self.occupancy_function = occupancy_function
        self.iso_level = iso_level
        self.bounds = bounds
        self.lower = bounds[0]
        self.upper = bounds[1]
        self.resolution = resolution
        self.mesh = None

        self.calculate_voxels()



    def calculate_voxel_matrix(self, max):

        """
        :param max:
        :return:
        """

        return

    def calculate_voxels(self):

        x_range = np.linspace(self.lower[0], self.upper[0], self.resolution[0])
        y_range = np.linspace(self.lower[1], self.upper[1], self.resolution[1])
        z_range = np.linspace(self.lower[2], self.upper[2], self.resolution[2])

        xx, yy, zz = np.meshgrid(x_range, y_range, z_range)
        xx = xx.flatten()
        yy = yy.flatten()
        zz = zz.flatten()

        points = np.array([xx, yy, zz]).reshape((xx.shape[0], 3))
        occupancy_mask = self.occupancy_function.evaluate_set(points)
        inside_points = points[occupancy_mask > self.iso_level]
        indices = points_to_indices(inside_points, pitch=1.0, origin=np.array([0.0, 0.0, 0.0]))

        self.mesh = points_to_marching_cubes(inside_points * 32, pitch=1.0)

