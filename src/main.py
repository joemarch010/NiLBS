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
from mesh.mesh_occupancy import MeshOccupancy

comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


bm_path = '../data/AMASS/body_models/smplh/female/model.npz'

bm = BodyModel(bm_path=bm_path).to(comp_device)
faces = c2c(bm.f)

npz_bdata_path = '../data/AMASS/MPILimits/MPI_Limits/03099/lar1_poses.npz' # the path to body data
bdata = np.load(npz_bdata_path)

print('Data keys available:%s'%list(bdata.keys()))
print('Vector poses has %d elements for each of %d frames.'%(bdata['poses'].shape[1], bdata['poses'].shape[0]))
print('Vector dmpls has %d elements for each of %d frames.'%(bdata['dmpls'].shape[1], bdata['dmpls'].shape[0]))
print('Vector trams has %d elements for each of %d frames.'%(bdata['trans'].shape[1], bdata['trans'].shape[0]))
print('Vector betas has %d elements constant for the whole sequence.'%bdata['betas'].shape[0])
print('The subject of the mocap sequence is %s.'%bdata['gender'])


fId = 100 # frame id of the mocap sequence

root_orient = torch.Tensor(bdata['poses'][fId:fId+1, :3]).to(comp_device) # controls the global root orientation
pose_body = torch.Tensor(bdata['poses'][fId:fId+1, 3:66]).to(comp_device) # controls the body
pose_hand = torch.Tensor(bdata['poses'][fId:fId+1, 66:]).to(comp_device) # controls the finger articulation


imw, imh=1000, 1000
mv = MeshViewer(width=imw, height=imh, use_offscreen=False)

body = bm(pose_body=pose_body)

weights = c2c(bm.weights)
#frameBonesOrigin = c2c(root_orient)
frameBonesOrigin = np.array([[0, 0, 0]])
frameBonesBody = c2c(pose_body)
frameBonesHand = c2c(pose_hand)
frameBones = np.append(frameBonesOrigin, frameBonesBody)
frameBones = np.append(frameBones, frameBonesHand)
boneHierachy = c2c(bm.kintree_table)
vertices = c2c(bm.v_template)[0]


wf = WeightingFunctionPointwise(vertices, weights);
md = LBSMeshDeformer(vertices, wf)
pose = pose_from_smplh(vertices, frameBones, boneHierachy, bm.J_regressor)
body_mesh = trimesh.Trimesh(vertices=md.apply_lbs(pose), faces=faces, vertex_colors=np.tile(colors['grey'], (6890, 1)))
ofm = OccupancyFunctionMesh(body_mesh)
psb = PointSamplerBox(body_mesh.bounds)
pss = PointSamplerSurface(body_mesh, noise='none', sigma=0.3)
#mo = MeshOccupancy(ofm, 0.5, body_mesh.bounds)
wts = WeightTrainSampler(ofm, body_mesh, weights)

weight_train_sample = wts.sample_pose(pose, 100, 100)


print('Occupancy mesh extracted')

n_points = 1024
sample_points = pss.sample_points(n_points)
occupancy_mask = ofm.evaluate_set(sample_points)
meshes = []

print('Sample points evaluated')

n_inside_points = 0
for i in range(0, n_points):

    if occupancy_mask[i] > 0:

        c = icosphere(2, 0.01)
        m = translation_matrix(sample_points[i])
        n_inside_points += 1
        c.apply_transform(m)

        #meshes.append(c)

#meshes.append(mo.mesh)
meshes.append(body_mesh)

print(n_inside_points)

mv.set_static_meshes(meshes)
mv.set_background_color([0.3, 0.4, 0.9])


# mv.render(render_wireframe=False)

mv.render(render_wireframe=False)

