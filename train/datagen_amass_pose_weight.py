
import numpy as np
import trimesh

from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c

from NiLBS.occupancy.occupancy_function_mesh import OccupancyFunctionMesh
from NiLBS.sampling.pose_sampler_amass import PoseSamplerAMASS
from NiLBS.sampling.weight_train_sampler import WeightTrainSampler
from NiLBS.skinning.skinning_mesh_lbs import LBSMeshDeformer
from NiLBS.weighting.weighting_function_pointwise import WeightingFunctionPointwise

if __name__ == '__main__':

    bm_path = '../data/AMASS/body_models/smplh/female/model.npz'
    bd_path = '../data/AMASS/MPILimits/MPI_Limits/03099/lar1_poses.npz'
    output_path = '../data/weight/weight_small_train.npz'
    n_frames = 100
    frame_step = 1
    frame_offset = 0

    bm = BodyModel(bm_path=bm_path)
    bdata = np.load(bd_path)

    faces = c2c(bm.f)
    weights = c2c(bm.weights)
    vertex_template = c2c(bm.v_template)[0]
    weighting_function = WeightingFunctionPointwise(vertex_template, weights)

    mesh_deformer = LBSMeshDeformer(vertex_template, weighting_function)

    psa = PoseSamplerAMASS(bm, bdata)
    poses = psa.sample_frames(n_frames=n_frames, step=frame_step, offset=frame_offset)
    sample_data = []

    for i in range(0, poses.shape[0]):

        pose = poses[i]
        posed_vertices = mesh_deformer.apply_lbs(pose)
        posed_mesh = trimesh.Trimesh(vertices=posed_vertices, faces=faces)
        posed_occupancy_function = OccupancyFunctionMesh(posed_mesh)
        weight_train_sampler = WeightTrainSampler(posed_occupancy_function, posed_mesh, weights)

        sample_data.append(weight_train_sampler.sample_pose(pose))

        print(str((i * 100.0) / poses.shape[0]) + '%')

    sample_data = np.array(sample_data)

    np.savez(output_path, sample_data)




