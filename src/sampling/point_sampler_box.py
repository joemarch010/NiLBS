
import numpy as np

class PointSamplerBox:
    """

    Class which generates sample points inside of a bounding box according to some distribution.

    @Note
        Currently only random uniform sampling is supported, but we may want to add more distributions in the future.


    """
    def __init__(self, box, distribution='uniform'):
        """
        :param box: Numpy array-like, axis aligned bounds of the box.
        :param distribution: String giving the distribution type, currently only 'uniform' is supported.
        """
        self.box = box
        self.distribution = distribution

        x1, y1, z1 = box[0, :]
        x2, y2, z2 = box[1, :]

        self.centre = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0, (z1 + z2) /2.0])
        self.scale = np.array([abs(x1 - x2), abs(y1 - y2), abs(z1 - z2)])

    def sample_points_uniform(self, n):

        top_left = np.array([self.centre[0] - self.scale[0] / 2.0,
                             self.centre[1] - self.scale[1] / 2.0,
                             self.centre[2] - self.scale[2] / 2.0])

        points = np.random.uniform(np.array([0, 0, 0]), self.scale, (n, 3))

        result = points + top_left

        return result

    def sample_points(self, n):
        """
        Return a collection of sample points within the bounding box and according to the distribution.

        :param n: int, number of points.
        :return: array-like, sampled points.
        """

        if self.distribution is 'uniform':
            return self.sample_points_uniform(n)

        return None