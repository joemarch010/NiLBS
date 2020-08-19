
class OccupancyFunction:

    def __init__(self):

        """

        Class representing an occupancy/indicator function.

        """

    def evaluate(self, x):

        """
        Evaluate the function at a single point
        :param x: x = (x, y, z), the point to evaluate the occupancy function for.
        :return: A value in the range [0, 1]. Usually this is is either 0.0 or 1.0, although in cases where the function
                is backed by a neural function, the return result will not be a discrete value.
        """

        pass

    def evaluate_set(self, X):
        """
        Evaluate the function at a set of points
        :param X: {(x, y, z)} (Numpy array-like)
        :return: {o | o in [0, 1]}
        """

        pass



