import numpy as np
import trimesh
from trimesh.creation import icosphere

from human_body_prior.tools.omni_tools import colors
from human_body_prior.mesh import MeshViewer
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.body_model.body_model import BodyModel

from NiLBS.occupancy.occupancy_function_mesh import OccupancyFunctionMesh
from NiLBS.occupancy.voxel.util import extract_voxel_grid


bm_path = '../data/AMASS/body_models/smplh/female/model.npz'
npz_bdata_path = '../data/AMASS/MPILimits/MPI_Limits/03099/lar1_poses.npz' # the path to body data
voxel_grid_file_path = '../data/voxel/test.npz'

bm = BodyModel(bm_path=bm_path)
faces = c2c(bm.f)

bdata = np.load(npz_bdata_path)

imw, imh=1000, 1000
mv = MeshViewer(width=imw, height=imh, use_offscreen=False)

weights = c2c(bm.weights)
vertices = c2c(bm.v_template)[0]

body_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))
ofm = OccupancyFunctionMesh(body_mesh)

extract_voxel_grid(ofm, body_mesh.bounds, np.array([32, 32, 32]), voxel_grid_file_path)

voxel_grid_file = np.load(voxel_grid_file_path)
voxel_grid = voxel_grid_file['voxel_grid']
voxel_start = voxel_grid_file['voxel_start']
voxel_dimensions = voxel_grid_file['voxel_dimensions']

print('Occupancy mesh extracted')

meshes = []

for i in range(0, voxel_grid.shape[0]):
    for j in range(0, voxel_grid.shape[1]):
        for k in range(0, voxel_grid.shape[2]):
            if voxel_grid[i][j][k] > 0.5:

                voxel_position = voxel_start + voxel_dimensions * np.array([i, j, k])

                box = trimesh.creation.box(voxel_dimensions, trimesh.transformations.translation_matrix(voxel_position))
                meshes.append(box)

print(len(meshes))

mv.set_static_meshes(meshes)
mv.set_background_color([0.3, 0.4, 0.9])

mv.render(render_wireframe=False)
