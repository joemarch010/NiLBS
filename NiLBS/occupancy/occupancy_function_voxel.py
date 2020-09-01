
from NiLBS.occupancy.occupancy_function import OccupancyFunction


class OccupancyFunctionVoxel(OccupancyFunction):
    """


    Occupancy function backed by a voxel grid.


    """
    def __init__(self, voxel_grid_file):

        self.voxel_grid = voxel_grid_file['voxel_gird']
        self.voxel_start = voxel_grid_file['voxel_start']
        self.voxel_dimensions = voxel_grid_file['voxel_dimensions']

    def evaluate(self, x):

        return None

    def evaluate_set(self, X):

        return None