
import numpy as np


class LBSMeshDeformer:
    """


    Deformer class which deforms weighted vertices according to some pose


    """

    def __init__(self, rest_vertices, weighting_function):
        """

        :param rest_vertices: Numpy array-like, Vx3
        :param weighting_function: WeightingFunction
        """
        self.rest_vertices = rest_vertices
        self.weighting_function = weighting_function

    def apply_lbs(self, pose):
        """

        Apply traditional LBS to the rest vertices according to some pose.

        :param pose: Pose, must have the same number of bones as self.weighting_function has weights
        :return: Numpy array-like, Vx3, the deformed vertices.
        """
        result = np.zeros(self.rest_vertices.shape)
        weights = self.weighting_function.evaluate_set(self.rest_vertices, pose)

        for i in range(0, self.rest_vertices.shape[0]):

            vertex = self.rest_vertices[i]
            vertex_homo = np.zeros((4, 1))
            vertex_homo[0] = vertex[0]
            vertex_homo[1] = vertex[1]
            vertex_homo[2] = vertex[2]
            vertex_homo[3] = 1

            vertex_weights = weights[i]
            bone_matrices = pose.bone_matrices
            transform_matrix = np.zeros((4, 4))

            for j in range(0, vertex_weights.shape[0]):

                transform_matrix += vertex_weights[j] * bone_matrices[j]

            vertex_trans = np.matmul(transform_matrix, vertex_homo)
            result[i] = vertex_trans[:3, 0]

        return result
