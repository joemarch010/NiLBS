import numpy as np
import trimesh
import pyrender


from NiLBS.weighting.weighting_function_pointwise import WeightingFunctionPointwise
from NiLBS.weighting.weighting_function_mlp_naive import WeightingFunctionMLPNaive
from NiLBS.weighting.weighting_function_mlp_rest_naive import WeightingFunctionMLPRestNaive
from NiLBS.demo.scene import make_default_model_pose_scene
from NiLBS.skinning.skinning_mesh_lbs import LBSDeformer
from NiLBS.skinning.util import redistribute_weights
from NiLBS.sampling.pose_sampler_amass import PoseSamplerAMASS
from NiLBS.body.human_body import HumanBody

bm_path = '../data/AMASS/body_models/smplh/female/model.npz'
npz_bdata_path = '../data/AMASS/MPILimits/MPI_Limits/03099/lar1_poses.npz'
weight_model_path_mlp = '../models/weight_naive'
weight_model_path_mlpr = '../models/weight_rest_naive'

body_model = HumanBody(body_dict_path=bm_path)
faces = body_model.faces

bdata = np.load(npz_bdata_path)
psa = PoseSamplerAMASS(body_model, bdata)
frame_poses = psa.sample_frames(step=100, n_frames=10)

imw, imh=1000, 1000

weights = body_model.weights
bone_hierarchy = body_model.bone_hierarchy
active_bones = range(0, 22)
weights = redistribute_weights(weights, bone_hierarchy, active_bones)
vertices = body_model.vertex_template

#wfmlp = WeightingFunctionMLPNaive(model_path=weight_model_path_mlp)
#wfmlpr = WeightingFunctionMLPRestNaive(model_path=weight_model_path_mlpr)
wf = WeightingFunctionPointwise(vertices, weights)

md_normal = LBSDeformer(wf)

#md_mlp = LBSMeshDeformer(vertices, wfmlp)
#md_mlpr = LBSMeshDeformer(vertices, wfmlpr)

body_mesh_normal = trimesh.Trimesh(vertices=md_normal.apply_lbs(vertices, frame_poses[2]), faces=faces, vertex_colors=np.tile((1.0, 1.0, 0.4), (6890, 1)))
#body_mesh_mlp = trimesh.Trimesh(vertices=md_mlp.apply_lbs(frame_poses[2]), faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))
#body_mesh_mlpr = trimesh.Trimesh(vertices=md_mlpr.apply_lbs(frame_poses[0]), faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))

print('Model Posed')

#body_mesh_mlp.apply_transform(trimesh.transformations.translation_matrix(np.array([2, 0, 0])))
#body_mesh_mlpr.apply_transform(trimesh.transformations.translation_matrix(np.array([-2, 0, 0])))

meshes = []

meshes.append(body_mesh_normal)
#meshes.append(body_mesh_mlp)
#meshes.append(body_mesh_mlpr)

print(len(meshes))


scene = pyrender.Scene()

mesh = pyrender.mesh.Mesh.from_trimesh(body_mesh_normal)
scene.add(mesh)

camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
camera_pose = np.eye(4)
camera_pose[:3, 3] = np.array([0, 0, 2.5])
scene.add(camera, pose=camera_pose)

scene = make_default_model_pose_scene()

viewer = pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=False)
viewer.run()
