import sys, os, time
import numpy as np
import trimesh
from trimesh.creation import icosphere
from trimesh.transformations import translation_matrix

from human_body_prior.tools.omni_tools import colors
from human_body_prior.mesh import MeshViewer
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.body_model.body_model import BodyModel

from NiLBS.occupancy.occupancy_function_mesh import OccupancyFunctionMesh
from NiLBS.occupancy.voxel.util import extract_voxel_grid
from NiLBS.weighting.weighting_function_pointwise import WeightingFunctionPointwise
from NiLBS.weighting.weighting_function_mlp_naive import WeightingFunctionMLPNaive
from NiLBS.weighting.weighting_function_mlp_rest_naive import WeightingFunctionMLPRestNaive
from NiLBS.pose.pose import Pose
from NiLBS.skinning.skinning_mesh_lbs import LBSDeformer
from NiLBS.pose.util import pose_from_smplh
from NiLBS.sampling.point_sampler_box import PointSamplerBox
from NiLBS.sampling.point_sampler_surface import PointSamplerSurface
from NiLBS.sampling.weight_train_sampler import WeightTrainSampler
from NiLBS.sampling.pose_sampler_amass import PoseSamplerAMASS
from NiLBS.mesh.mesh_occupancy import MeshOccupancy


bm_path = '../data/AMASS/body_models/smplh/female/model.npz'

bm = BodyModel(bm_path=bm_path)
faces = c2c(bm.f)

npz_bdata_path = '../data/AMASS/MPILimits/MPI_Limits/03099/lar1_poses.npz' # the path to body data
bdata = np.load(npz_bdata_path)

psa = PoseSamplerAMASS(bm, bdata)
frame_poses = psa.sample_frames(step=100, n_frames=10)

imw, imh=1000, 1000
mv = MeshViewer(width=imw, height=imh, use_offscreen=False)


weights = c2c(bm.weights)
vertices = c2c(bm.v_template)[0]


weight_model_path = '../models/weight_rest_naive'

#wfmlp = WeightingFunctionMLPNaive(model_path=weight_model_path)
wfmlpr = WeightingFunctionMLPRestNaive(model_path=weight_model_path)
wf = WeightingFunctionPointwise(vertices, weights)

md = LBSDeformer(vertices, wfmlpr)
body_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))
ofm = OccupancyFunctionMesh(body_mesh)

#extract_voxel_grid(ofm, body_mesh.bounds, np.array([32, 32, 32]), '../data/voxel/test.npz')

voxel_grid_file = np.load('../data/voxel/test.npz')
voxel_grid = voxel_grid_file['voxel_grid']
voxel_start = voxel_grid_file['voxel_start']
voxel_dimensions = voxel_grid_file['voxel_dimensions']

#mo = MeshOccupancy(ofm, 0.5, body_mesh.bounds)
wts = WeightTrainSampler(ofm, body_mesh, weights)

print('Occupancy mesh extracted')

meshes = []

for i in range(0, voxel_grid.shape[0]):
    for j in range(0, voxel_grid.shape[1]):
        for k in range(0, voxel_grid.shape[2]):
            if voxel_grid[i][j][k] > 0.5:

                voxel_position = voxel_start + voxel_dimensions * np.array([i, j, k])

                box = trimesh.creation.box(voxel_dimensions, trimesh.transformations.translation_matrix(voxel_position))
                meshes.append(box)


#meshes.append(body_mesh)

print(len(meshes))

mv.set_static_meshes(meshes)
mv.set_background_color([0.3, 0.4, 0.9])


mv.render(render_wireframe=False)

