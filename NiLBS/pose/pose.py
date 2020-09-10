
import numpy as np


class Pose:

    def __init__(self, bone_matrices):
        """

        Create bose from raw bone matrices.

        :param bone_matrices: Numpy array-like, Bx4x4 array of bone transformation matrices.
        """
        self.bone_matrices = bone_matrices
        self.inverse_bone_matrices = np.zeros(bone_matrices.shape)
        self.posed_joints = np.zeros((bone_matrices.shape[0], 3))

        for i in range(0, bone_matrices.shape[0]):
            self.inverse_bone_matrices[i] = np.linalg.inv(bone_matrices[i])

        for i in range(0, bone_matrices.shape[0]):
            self.posed_joints[i][0] = self.bone_matrices[i][0][3]
            self.posed_joints[i][1] = self.bone_matrices[i][1][3]
            self.posed_joints[i][0] = self.bone_matrices[i][2][3]

    def get_naive_encoding(self, x):
        """

        :return: Numpy array-like, flattened bone matrix containing B * 4 * 4 elements.
        """
        return self.bone_matrices.flatten()

    def get_nasa_encoding(self, x):
        """

        Produces a pose encoding identical to the one used in the NASA paper.

        :param x: Numpy array-like, 3x1, translation vector of point to be queried.
        :return: Numpy array-like, B * 3, the product of each bone multiplied by the root translation vector and then
                 de-homogenised.
        """
        result = np.zeros(3 * self.bone_matrices.shape[0])
        x_homo = np.zeros((4, 1))
        x_homo[0] = x[0]
        x_homo[1] = x[1]
        x_homo[2] = x[2]

        for i in range(0, self.inverse_bone_matrices.shape[0]):

            x_trans = np.matmul(self.inverse_bone_matrices[i], x_homo)
            result[i * 3 + 0] = x_trans[0]
            result[i * 3 + 1] = x_trans[1]
            result[i * 3 + 2] = x_trans[2]

        return result

    def get_nilbs_encoding(self, x):
        """

        Produces the pose encoding suggested in the NiLBS Technical Report: NASA encoding with an additional ghost bone.

        :param x: Numpy array-like, 3x1, translation vector of point to be queried
        :return: Numpy array-like, (B + 1) * 3, the product of each bone multiplied by the root translation vector and then
                 de-homogenised.
        """
        nasa_encoding = self.get_nasa_encoding(x)
        ghost_bone_corrective = nasa_encoding[0:3]

        result = np.concatenate((nasa_encoding, ghost_bone_corrective), axis=0)

        return result
