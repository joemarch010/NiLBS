
import pyrender
import trimesh
import numpy as np
import os

from NiLBS.body.human_body import HumanBody
from NiLBS.sampling.pose_sampler_amass import PoseSamplerAMASS
from NiLBS.skinning.skinning_mesh_lbs import LBSDeformer
from NiLBS.weighting.weighting_function_pointwise import WeightingFunctionPointwise


default_bm_path = os.path.join(os.path.dirname(__file__), '../data/female_body_model_smplh.npz')
default_pose_path = os.path.join(os.path.dirname(__file__), '../data/amass_mpi_sample_poses.npz')

body_model = HumanBody(body_dict_path=default_bm_path, active_bones=range(0, 22))
pose_sampler = PoseSamplerAMASS(body_model, np.load(default_pose_path))
poses = pose_sampler.sample_frames(10)
pose = poses[0]


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