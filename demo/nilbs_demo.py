import numpy as np
import trimesh
import copy

from human_body_prior.tools.omni_tools import colors
from human_body_prior.mesh import MeshViewer
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.body_model.body_model import BodyModel

from NiLBS.weighting.weighting_function import WeightingFunction
from NiLBS.skinning.util import redistribute_weights
from NiLBS.sampling.pose_sampler_amass import PoseSamplerAMASS
from NiLBS.occupancy.occupancy_function_mesh import OccupancyFunctionMesh
from NiLBS.occupancy.occupancy_function_rest_cached import OccupancyFunctionRestCached
from NiLBS.occupancy.voxel.util import extract_voxel_grid


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
wf = WeightingFunction(weights.shape[1] + 1)
body_mesh_rest = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))
rest_occupancy = OccupancyFunctionMesh(body_mesh_rest)
posed_occupancy = OccupancyFunctionRestCached(rest_occupancy, frame_poses[0], wf)
voxel_bounds = copy.copy(body_mesh_rest.bounds)
t = voxel_bounds[0][1]
voxel_bounds[0][1] = voxel_bounds[0][2] * 2
voxel_bounds[0][2] = t
voxel_bounds[0][0] *= 2
t = voxel_bounds[1][1]
voxel_bounds[1][1] = voxel_bounds[1][2] * 2
voxel_bounds[1][2] = t
voxel_bounds[1][0] *= 2
voxel_dict = extract_voxel_grid(posed_occupancy, voxel_bounds, np.array([32, 32, 32]))
voxel_grid = voxel_dict['voxel_grid']
voxel_start = voxel_dict['voxel_start']
voxel_dimensions = voxel_dict['voxel_dimensions']

print('Iso-Surface Extracted')

meshes = []

for i in range(0, voxel_grid.shape[0]):
    for j in range(0, voxel_grid.shape[1]):
        for k in range(0, voxel_grid.shape[2]):
            if voxel_grid[i][j][k] > 0.5:

                voxel_position = voxel_start + voxel_dimensions * np.array([i, j, k])

                box = trimesh.creation.box(voxel_dimensions, trimesh.transformations.translation_matrix(voxel_position))
                meshes.append(box)

print('Meshes Added')

print(len(meshes))

mv.set_static_meshes(meshes)
mv.set_background_color([0.3, 0.4, 0.9])

mv.render(render_wireframe=False)
