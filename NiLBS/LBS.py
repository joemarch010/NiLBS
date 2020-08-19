
import numpy as np
import torch
from human_body_prior.tools.omni_tools import copy2cpu as c2c


def rodrigues(rotations):

    nRotations = int(rotations.shape[0] / 3)
    result = np.zeros((nRotations, 3, 3))

    for i in range(0, rotations.shape[0], 3):

        rotation = np.array([rotations[i], rotations[i + 1], rotations[i + 2]])
        angle = np.linalg.norm(rotation)
        axis = rotation / (angle + 0.0001)
        A = np.zeros((3, 3))
        cosAngle = np.cos(angle)
        sinAngle = np.sin(angle)

        A[0][1] = -axis[2]
        A[0][2] = axis[1]
        A[1][0] = axis[2]
        A[1][2] = -axis[0]
        A[2][0] = -axis[1]
        A[2][1] = axis[0]

        result[int(i / 3)] = np.identity(3) + (sinAngle * A) + ((1 - cosAngle) * np.matmul(A, A))

    return result


def calculateRotationMatrices(bones, boneHierachy):

    matrices = rodrigues(bones)

    return matrices


def vertices2joints(vertices, J_regressor):
    v_torch = torch.tensor(vertices)
    return torch.einsum('ik,ji->jk', [v_torch, J_regressor])


def transform_matrix(rotation, translation):

    rotation_mat = np.pad(rotation, pad_width=((0, 1), (0, 1)), constant_values=0)
    translation = np.reshape(translation, (3, 1))
    translation_mat = np.pad(translation, ((0, 1), (3, 0)), constant_values=0)
    translation_mat[3][3] = 1

    result = rotation_mat + translation_mat

    return result


def get_rigid_transforms(rotation_matrices, joints, bone_hierarchy):

    relative_joints = np.zeros(joints.shape)
    relative_joints[0] = joints[0]

    for i in range(1, bone_hierarchy.shape[1]):

        relative_joints[i] = joints[i] - joints[bone_hierarchy[0][i]]

    transform_matrices = np.zeros((bone_hierarchy.shape[1], 4, 4))
    transform_matrices[0] = transform_matrix(rotation_matrices[0], relative_joints[0])

    for i in range(1, bone_hierarchy.shape[1]):

        transform_matrices[i] = np.matmul(transform_matrices[bone_hierarchy[0][i]], transform_matrix(rotation_matrices[i], relative_joints[i]))

    for i in range(0, joints.shape[0]):

        joint_homo = np.zeros((4, 1))
        joint_homo[0] = joints[i][0]
        joint_homo[1] = joints[i][1]
        joint_homo[2] = joints[i][2]

        joint_trans = np.matmul(transform_matrices[i], joint_homo)

        transform_matrices[i][0][3] -= joint_trans[0]
        transform_matrices[i][1][3] -= joint_trans[1]
        transform_matrices[i][2][3] -= joint_trans[2]

    return transform_matrices


def LBS(vertices, weights, bones, bone_hierarchy, J_regressor):

    rotationMatrices = calculateRotationMatrices(bones, bone_hierarchy)
    J = vertices2joints(vertices, J_regressor)
    joints = c2c(J)
    result = np.zeros(vertices.shape)
    transform_matrices = get_rigid_transforms(rotationMatrices, joints, bone_hierarchy)

    for i in range(0, vertices.shape[0]):
        vertex = vertices[i]
        vertex_homo = np.zeros((4, 1))
        vertex_homo[0] = vertex[0]
        vertex_homo[1] = vertex[1]
        vertex_homo[2] = vertex[2]
        vertex_homo[3] = 1

        translation_matrix = np.zeros((4, 4))

        for j in range(0, transform_matrices.shape[0]):

            translation_matrix += weights[i][j] * transform_matrices[j]

        result_homo = np.matmul(translation_matrix, vertex_homo)
        result[i] = result_homo[:3, 0]

    return result