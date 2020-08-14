
import numpy as np


class PointSamplerSurface:
    """


    Class which generates points sampled from locations on the surfaces of a mesh.


    """
    def __init__(self, mesh, distribution='uniform', noise='none', sigma=0):
        """
        :param mesh: Mesh, must have instance variables 'vertices' and 'faces'
        :param distribution: String, must be one of {'uniform', 'poisson'}
        :param noise: String, must be one of {'none', 'isotropic'}
        :param sigma: Float, used when calculating noise
        """
        self.mesh = mesh
        self.distribution = distribution
        self.noise = noise
        self.sigma = sigma

    def generate_isptropic_noise_samples(self, n):

        result = np.zeros(n, 3)

        return result

    def generate_noise_samples(self, n):

        if self.noise is 'isotropic':

            return self.generate_isptropic_noise_samples(n)
        else:

            return np.zeros(n, 3)

    def sample_points_uniform(self, n):

        result = np.zeros((n, 3))

        for i in range(0, n):

            tri_index = np.random.randint(0, self.mesh.triangles.shape[0])
            c1 = np.random.uniform(0.0, 1.0)
            c2 = np.random.uniform(0.0, 1.0)
            c3 = np.random.uniform(0.0, 1.0)

            v1 = self.mesh.triangles[tri_index][0]
            v2 = self.mesh.triangles[tri_index][1]
            v3 = self.mesh.triangles[tri_index][2]
            v = c1 * v1 + c2 * v2 + c3 *v3

            result[i] = v

        return result

    def sample_points(self, n):

        if self.distribution is 'uniform':

            return self.sample_points_uniform(n)
        else:

            return np.zeros((n, 3))