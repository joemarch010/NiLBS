
import numpy as np

from occupancy.occupancy_function import OccupancyFunction
from occupancy.winding_numbers import is_inside_turbo


class OccupancyFunctionMesh(OccupancyFunction):

    def __init__(self, mesh):
        """
        :param mesh: The mesh for use for use when evaluating the function. Note that mesh must contain both
                    vertex, 'vertices' and face, 'faces' information.
        """

        self.mesh = mesh


    def evaluate(self, x):
        """
        Evaluate the occupancy function using a fast winding numbers approach

        :param x: (x, y, z)
        :return: a value in {0, 1}.
        """
        input = np.zeros((1, 3))
        input[0][0] = x[0]
        input[0][1] = x[1]
        input[0][2] = x[2]

        if is_inside_turbo(self.mesh.triangles, input):

            return 1.0
        else:

            return 0.0