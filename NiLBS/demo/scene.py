
import pyrender
import trimesh
import numpy as np
import os

from NiLBS.body.human_body import HumanBody
from NiLBS.occupancy.occupancy_function_mesh import OccupancyFunctionMesh
from NiLBS.occupancy.occupancy_function_rest_cached import OccupancyFunctionRestCached
from NiLBS.occupancy.voxel.util import extract_voxel_grid
from NiLBS.sampling.pose_sampler_amass import PoseSamplerAMASS
from NiLBS.skinning.skinning_mesh_lbs import LBSDeformer
from NiLBS.weighting.weighting_function import WeightingFunction
from NiLBS.weighting.weighting_function_pointwise import WeightingFunctionPointwise
from NiLBS.weighting.weighting_function_nearest_neighbour import WeightingFunctionNearestNeighbour


default_bm_path = os.path.join(os.path.dirname(__file__), '../data/female_body_model_smplh.npz')
default_pose_path = os.path.join(os.path.dirname(__file__), '../data/amass_mpi_sample_poses.npz')

body_model = HumanBody(body_dict_path=default_bm_path, active_bones=range(0, 22))
pose_sampler = PoseSamplerAMASS(body_model, np.load(default_pose_path))
poses = pose_sampler.sample_frames(10, step=50)
pose = poses[2]


def make_default_model_pose_scene():
    """
`

    :return:
    """

    result = pyrender.Scene()
    weighting_function = WeightingFunctionPointwise(body_model.vertex_template, body_model.weights)
    mesh_deformer = LBSDeformer(weighting_function)
    deformed_vertices = mesh_deformer.apply_lbs(body_model.vertex_template, pose)
    trimesh_mesh = trimesh.Trimesh(vertices=deformed_vertices, faces=body_model.faces,
                                       vertex_colors=np.tile((1.0, 1.0, 0.4), (6890, 1)))
    mesh = pyrender.Mesh.from_trimesh(trimesh_mesh)
    result.add(mesh)

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = np.array([0, 0, 2.5])
    result.add(camera, pose=camera_pose)

    return result


def make_default_joint_view_scene():
    """
`

    :return:
    """

    result = pyrender.Scene()

    for i in range(0, pose.bone_matrices.shape[0]):

        joint = body_model.joints[i]
        joint_homo = np.zeros((4, 1))
        joint_homo[0] = joint[0]
        joint_homo[1] = joint[1]
        joint_homo[2] = joint[2]
        joint_trans = np.matmul(pose.bone_matrices[i], joint_homo)
        joint_pos = joint_trans[0:3, 0]

        trimesh_mesh = trimesh.creation.box(np.array([0.1, 0.1, 0.1]),
                                            trimesh.transformations.translation_matrix(joint_pos),
                                            vertex_colors=np.tile((1.0, 0, 0), (8, 1)))
        box_mesh = pyrender.Mesh.from_trimesh(trimesh_mesh)
        result.add(box_mesh)

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = np.array([0, 0, 2.5])
    result.add(camera, pose=camera_pose)

    return result


def make_default_nilbs_scene():

    result = pyrender.Scene()
    posed_joints = np.zeros((pose.bone_matrices.shape[0], 3))

    for i in range(0, pose.bone_matrices.shape[0]):

        joint = body_model.joints[i]
        joint_homo = np.zeros((4, 1))
        joint_homo[0] = joint[0]
        joint_homo[1] = joint[1]
        joint_homo[2] = joint[2]
        joint_trans = np.matmul(pose.bone_matrices[i], joint_homo)
        joint_pos = joint_trans[0:3, 0]
        posed_joints[i] = joint_pos


    weighting_function = WeightingFunctionNearestNeighbour(posed_joints)
    rest_trimesh_mesh = trimesh.Trimesh(vertices=body_model.vertex_template, faces=body_model.faces,
                                        vertex_colors=np.tile((1.0, 1.0, 0.4), (6890, 1)))
    rest_occupancy_function = OccupancyFunctionMesh(rest_trimesh_mesh)
    posed_occupancy_function = OccupancyFunctionRestCached(rest_occupancy_function, pose, weighting_function)
    voxel_dict = extract_voxel_grid(posed_occupancy_function, rest_trimesh_mesh.bounds, np.array([32, 32, 32]))
    voxel_grid = voxel_dict['voxel_grid']
    voxel_start = voxel_dict['voxel_start']
    voxel_dimensions = voxel_dict['voxel_dimensions']


    for i in range(0, voxel_grid.shape[0]):
        for j in range(0, voxel_grid.shape[1]):
            for k in range(0, voxel_grid.shape[2]):
                if voxel_grid[i][j][k] > 0.5:
                    voxel_position = voxel_start + voxel_dimensions * np.array([i, j, k])

                    trimesh_mesh = trimesh.creation.box(voxel_dimensions,
                                                        trimesh.transformations.translation_matrix(voxel_position),
                                                        vertex_colors=np.tile((1.0, 1.0, 0.4), (8, 1)))
                    box_mesh = pyrender.Mesh.from_trimesh(trimesh_mesh)
                    result.add(box_mesh)

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = np.array([0, 0, 2.5])
    result.add(camera, pose=camera_pose)

    return result