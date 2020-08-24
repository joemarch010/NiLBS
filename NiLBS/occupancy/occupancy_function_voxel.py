
from NiLBS.occupancy.occupancy_function import OccupancyFunction


class OccupancyFunctionVoxel(OccupancyFunction):
    """


    Occupancy function backed by a voxel grid.


    """
    def __init__(self, voxel_grid):

        self.voxel_grid = voxel_grid
        