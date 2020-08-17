
import numpy as np


class Pose:

    def __init__(self, bone_matrices):
        """

        Create bose from raw bone matrices.

        :param bone_matrices: Numpy array-like, Bx4x4 array of bone transformation matrices.
        """
        self.bone_matrices = bone_matrices

    def get_naive_encoding(self):
        """

        :return: Numpy array-like, flattened bone matrix containing B * 4 * 4 elements.
        """
        return self.bone_matrices.flatten()

    def get_nasa_encoding(self, t0):
        """

        Produces a pose encoding identical to the one used in the NASA paper.

        :param t0: Numpy array-like, 3x1, translation vector of root bone.
        :return: Numpy array-like, B * 3, the product of each bone multipled by the root translation vector and then
                 de-homogenised.
        """
        result = np.zeros(3 * self.bone_matrices.shape[0])
        t0_homo = np.zeros((4, 1))
        t0_homo[0] = t0[0]
        t0_homo[1] = t0[1]
        t0_homo[2] = t0[2]

        for i in range(0, self.bone_matrices.shape[0]):

            t0_trans = np.matmul(self.bone_matrices[i], t0_homo)
            result[i * 3 + 0] = t0_trans[0]
            result[i * 3 + 1] = t0_trans[1]
            result[i * 3 + 2] = t0_trans[2]

        return result
