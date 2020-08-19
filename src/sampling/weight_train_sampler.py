
import numpy as np

from sampling.point_sampler_box import PointSamplerBox
from sampling.point_sampler_surface import PointSamplerSurface
from skinning.skinning_mesh_lbs import LBSMeshDeformer
from weighting.weighting_function_pointwise import WeightingFunctionPointwise


class WeightTrainSampler:

    def __init__(self, occupancy_function, mesh, weights):

        self.occupancy_function = occupancy_function
        self.mesh = mesh
        self.weights = weights

        wf = WeightingFunctionPointwise(mesh.vertices, weights)
        self.lbs_deformer = LBSMeshDeformer(mesh.vertices, wf)

    def sample_pose(self, pose, n_bb_points=1024, n_surface_points=1024):

        posed_vertices = self.lbs_deformer.apply_lbs(pose)
        result = dict()
        psb = PointSamplerBox(self.mesh.bounds)
        pss = PointSamplerSurface(self.mesh, noise='isotropic', sigma=0.3)

        bb_points = psb.sample_points(n_bb_points)
        surface_points = pss.sample_points(n_surface_points)
        bb_occupancy = np.reshape(self.occupancy_function.evaluate_set(bb_points), (n_bb_points, 1))
        surface_occupancy = np.reshape(self.occupancy_function.evaluate_set(surface_points), (n_surface_points, 1))

        bb_occupancy_points = np.concatenate((bb_points, bb_occupancy), axis=1)
        surface_occupancy_points = np.concatenate((surface_points, surface_occupancy), axis=1)
        occupancy_points = np.concatenate((bb_occupancy_points, surface_occupancy_points), axis=0)

        result['vertices'] = posed_vertices
        result['weights'] = self.weights
        result['occupancy_points'] = occupancy_points

        return result
