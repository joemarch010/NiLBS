import sys, os, time
import torch
import numpy as np
import trimesh
from trimesh.creation import icosphere
from trimesh.transformations import translation_matrix

from human_body_prior.tools.omni_tools import colors
from human_body_prior.mesh import MeshViewer
from human_body_prior.mesh.sphere import points_to_spheres
from LBS import LBS
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.body_model.body_model import BodyModel

from occupancy.occupancy_function_mesh import OccupancyFunctionMesh
from weighting.weighting_function_pointwise import WeightingFunctionPointwise
from pose.pose import Pose
from skinning.skinning_mesh_lbs import LBSMeshDeformer
from pose.util import pose_from_smplh
from sampling.point_sampler_box import PointSamplerBox
from sampling.point_sampler_surface import PointSamplerSurface
from sampling.weight_train_sampler import WeightTrainSampler
from sampling.pose_sampler_amass import PoseSamplerAMASS
from mesh.mesh_occupancy import MeshOccupancy

comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


bm_path = '../data/AMASS/body_models/smplh/female/model.npz'

bm = BodyModel(bm_path=bm_path).to(comp_device)
faces = c2c(bm.f)

npz_bdata_path = '../data/AMASS/MPILimits/MPI_Limits/03099/lar1_poses.npz' # the path to body data
bdata = np.load(npz_bdata_path)

psa = PoseSamplerAMASS(bm, bdata)
frame_poses = psa.sample_frames(step=100)

imw, imh=1000, 1000
mv = MeshViewer(width=imw, height=imh, use_offscreen=False)


weights = c2c(bm.weights)
vertices = c2c(bm.v_template)[0]


wf = WeightingFunctionPointwise(vertices, weights);
md = LBSMeshDeformer(vertices, wf)
body_mesh = trimesh.Trimesh(vertices=md.apply_lbs(frame_poses[1]), faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))
ofm = OccupancyFunctionMesh(body_mesh)
#mo = MeshOccupancy(ofm, 0.5, body_mesh.bounds)
wts = WeightTrainSampler(ofm, body_mesh, weights)

weight_train_samples = []

for i in range(0, frame_poses.shape[0]):

    weight_train_samples.append(wts.sample_pose(frame_poses[i]))

print('Occupancy mesh extracted')

meshes = []

meshes.append(body_mesh)

mv.set_static_meshes(meshes)
mv.set_background_color([0.3, 0.4, 0.9])


mv.render(render_wireframe=False)

