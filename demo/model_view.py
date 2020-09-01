import numpy as np
import trimesh

from human_body_prior.tools.omni_tools import colors
from human_body_prior.mesh import MeshViewer
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.body_model.body_model import BodyModel

from NiLBS.weighting.weighting_function_pointwise import WeightingFunctionPointwise
from NiLBS.weighting.weighting_function_mlp_naive import WeightingFunctionMLPNaive
from NiLBS.weighting.weighting_function_mlp_rest_naive import WeightingFunctionMLPRestNaive
from NiLBS.skinning.skinning_mesh_lbs import LBSDeformer
from NiLBS.skinning.util import redistribute_weights
from NiLBS.sampling.pose_sampler_amass import PoseSamplerAMASS


bm_path = '../data/AMASS/body_models/smplh/female/model.npz'
npz_bdata_path = '../data/AMASS/MPILimits/MPI_Limits/03099/lar1_poses.npz'
weight_model_path_mlp = '../models/weight_naive'
weight_model_path_mlpr = '../models/weight_rest_naive'

bm = BodyModel(bm_path=bm_path)
faces = c2c(bm.f)

bdata = np.load(npz_bdata_path)
psa = PoseSamplerAMASS(bm, bdata)
frame_poses = psa.sample_frames(step=100, n_frames=10)

imw, imh=1000, 1000
mv = MeshViewer(width=imw, height=imh, use_offscreen=False)

weights = c2c(bm.weights)
bone_hierarchy = c2c(bm.kintree_table)
active_bones = range(0, 22)
weights = redistribute_weights(weights, bone_hierarchy, active_bones)
vertices = c2c(bm.v_template)[0]

#wfmlp = WeightingFunctionMLPNaive(model_path=weight_model_path_mlp)
#wfmlpr = WeightingFunctionMLPRestNaive(model_path=weight_model_path_mlpr)
wf = WeightingFunctionPointwise(vertices, weights)

md_normal = LBSDeformer(wf)

#md_mlp = LBSMeshDeformer(vertices, wfmlp)
#md_mlpr = LBSMeshDeformer(vertices, wfmlpr)

body_mesh_normal = trimesh.Trimesh(vertices=md_normal.apply_lbs(vertices, frame_poses[2]), faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))
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

mv.set_static_meshes(meshes)
mv.set_background_color([0.3, 0.4, 0.9])


mv.render(render_wireframe=False)
