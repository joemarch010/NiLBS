
import numpy as np


class VoxelTree:
    """


    Tree-Based representation of a sparse voxel grid intended to keep memory consumption down to a reasonable level.

    @Note
        This is in not a k-d tree.


    """

    class VoxelTreeNode:

        def __init__(self, bounds, resolution):

            self.bounds = bounds
            self.resolution = resolution
            self.children = []

    class VoxelTreeLeafNode(VoxelTreeNode):

        def __init__(self, occupancy_function, bounds, resolution):
            super.__init__(bounds, resolution)



    def __init__(self, occupancy_function, bounds, initial_resolution, maximum_resolution):

        self.occupancy_function = occupancy_function
        self.bounds = bounds
        self.initial_resolution = initial_resolution
        self.maximum_resolution = maximum_resolution

        bound_dimensions = np.abs(bounds[0] - bounds[1])

        self.minimum_voxel_dimensions = bound_dimensions / maximum_resolution

