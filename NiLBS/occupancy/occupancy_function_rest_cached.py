
from NiLBS.occupancy.occupancy_function import OccupancyFunction

class OccupancyFunctionRestCached(OccupancyFunction):
    """

    Occupancy function backed by an OccupancyFunction at rest, a Pose, and a WeightingFunction.


    """
    def __init__(self, rest_occupancy_function, weighting_function):

        self.rest_occupancy_function = rest_occupancy_function
        self.weighting_function = weighting_function