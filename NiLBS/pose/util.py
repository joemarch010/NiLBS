
import numpy as np

from NiLBS.pose.pose import Pose


#def calculate_joints(vertices, j_regressor):

#    vertex_tensor = torch.tensor(vertices)
#    return torch.einsum('ik,ji->jk', [vertex_tensor, j_regressor])


def rodrigues(rotations):

    n_rotations = int(rotations.shape[0] / 3)
    result = np.zeros((n_rotations, 3, 3))

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


def transform_matrix(rotation, translation):

    rotation_mat = np.pad(rotation, pad_width=((0, 1), (0, 1)), constant_values=0)
    translation = np.reshape(translation, (3, 1))
    translation_mat = np.pad(translation, ((0, 1), (3, 0)), constant_values=0)
    translation_mat[3][3] = 1

    result = rotation_mat + translation_mat

    return result


def calculate_bone_matrices_smplh(rotation_matrices, joints, bone_hierarchy):

    relative_joints = np.zeros(joints.shape)
    relative_joints[0] = joints[0]

    for i in range(1, rotation_matrices.shape[0]):

        relative_joints[i] = joints[i] - joints[bone_hierarchy[0][i]]

    transform_matrices = np.zeros((rotation_matrices.shape[0], 4, 4))
    transform_matrices[0] = transform_matrix(rotation_matrices[0], relative_joints[0])

    for i in range(1, rotation_matrices.shape[0]):

        transform_matrices[i] = np.matmul(transform_matrices[bone_hierarchy[0][i]], transform_matrix(rotation_matrices[i], relative_joints[i]))

    for i in range(0, rotation_matrices.shape[0]):

        joint_homo = np.zeros((4, 1))
        joint_homo[0] = joints[i][0]
        joint_homo[1] = joints[i][1]
        joint_homo[2] = joints[i][2]

        joint_trans = np.matmul(transform_matrices[i], joint_homo)

        transform_matrices[i][0][3] -= joint_trans[0]
        transform_matrices[i][1][3] -= joint_trans[1]
        transform_matrices[i][2][3] -= joint_trans[2]

    return transform_matrices


def pose_from_smplh(vertex_template, full_pose, bone_hierarchy, joints):
    """

    Produces a pose from the SMPLH configuration provided.

    :param vertex_template: Numpy array-like, V x 3, vertex position data at rest.
    :param full_pose: Numpy array-like, B x 3, bone rotation data.
    :param bone_hierarchy: Numpy array-like, B, index of parent bone, -1 if root.
    :param joints
    :return: Fully configured Pose which ban be used on any mesh with the same vertex template.
    """

    rotation_matrices = rodrigues(full_pose)
    bone_matrices = calculate_bone_matrices_smplh(rotation_matrices, joints, bone_hierarchy)
    result = Pose(bone_matrices)

    return result