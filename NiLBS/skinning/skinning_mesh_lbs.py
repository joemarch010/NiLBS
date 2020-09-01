
import numpy as np


class LBSDeformer:
    """


    Deformer class which deforms weighted vertices according to some pose


    """

    def __init__(self, weighting_function):
        """

        :param weighting_function: WeightingFunction
        """
        self.weighting_function = weighting_function

    def apply_lbs(self, rest_vertices, pose):
        """


        Apply traditional LBS to the rest vertices according to some pose.

        :param pose: Pose, must have the same number of bones as self.weighting_function has weights
        :return: Numpy array-like, Vx3, the deformed vertices.
        """
        result = np.zeros(rest_vertices.shape)
        weights = self.weighting_function.evaluate_set(rest_vertices, pose)

        for i in range(0, rest_vertices.shape[0]):

            vertex = rest_vertices[i]
            vertex_homo = np.zeros((4, 1))
            vertex_homo[0] = vertex[0]
            vertex_homo[1] = vertex[1]
            vertex_homo[2] = vertex[2]
            vertex_homo[3] = 1

            vertex_weights = weights[i]
            bone_matrices = pose.bone_matrices
            transform_matrix = np.zeros((4, 4))

            for j in range(0, pose.bone_matrices.shape[0]):

                transform_matrix += vertex_weights[j] * bone_matrices[j]

            vertex_trans = np.matmul(transform_matrix, vertex_homo)
            result[i] = vertex_trans[:3, 0]

        return result

    def invert_lbs(self, rest_vertices, pose):
        """



        :param pose:
        :return:
        """
        result = np.zeros(rest_vertices.shape)
        weights = self.weighting_function.evaluate_set(rest_vertices, pose)

        for i in range(0, rest_vertices.shape[0]):

            vertex = rest_vertices[i]
            vertex_homo = np.zeros((4, 1))
            vertex_homo[0] = vertex[0]
            vertex_homo[1] = vertex[1]
            vertex_homo[2] = vertex[2]
            vertex_homo[3] = 1

            vertex_weights = weights[i]
            bone_matrices = pose.inverse_bone_matrices
            transform_matrix = np.zeros((4, 4))

            for j in range(0, pose.inverse_bone_matrices.shape[0]):

                transform_matrix += vertex_weights[j] * bone_matrices[j]

            vertex_trans = np.matmul(transform_matrix, vertex_homo)
            result[i] = vertex_trans[:3, 0]

        return result


