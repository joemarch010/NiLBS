import numpy as np
import trimesh

from human_body_prior.tools.omni_tools import colors
from human_body_prior.mesh import MeshViewer
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.body_model.body_model import BodyModel

from NiLBS.weighting.weighting_function_pointwise import WeightingFunctionPointwise
from NiLBS.skinning.skinning_mesh_lbs import LBSDeformer
from NiLBS.skinning.util import redistribute_weights
from NiLBS.sampling.pose_sampler_amass import PoseSamplerAMASS


bm_path = '../data/AMASS/body_models/smplh/female/model.npz'
npz_bdata_path = '../data/AMASS/MPILimits/MPI_Limits/03099/lar1_poses.npz'

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
wf = WeightingFunctionPointwise(vertices, weights)
md_normal = LBSDeformer(wf)
deformed_vertices = md_normal.apply_lbs(vertices, frame_poses[0])
deformed_wf = WeightingFunctionPointwise(deformed_vertices, weights)
md_inverse = LBSDeformer(deformed_wf)

body_mesh_normal = trimesh.Trimesh(vertices=md_inverse.invert_lbs(deformed_vertices, frame_poses[0]), faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))
print('Model Posed')


meshes = []

meshes.append(body_mesh_normal)

print(len(meshes))

mv.set_static_meshes(meshes)
mv.set_background_color([0.3, 0.4, 0.9])


mv.render(render_wireframe=False)
