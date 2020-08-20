
import numpy as np

from NiLBS.sampling.point_sampler_box import PointSamplerBox
from NiLBS.sampling.point_sampler_surface import PointSamplerSurface
from NiLBS.skinning.skinning_mesh_lbs import LBSMeshDeformer
from NiLBS.weighting.weighting_function_pointwise import WeightingFunctionPointwise


class WeightTrainSampler:
    """


    Class for creating sample data for use in training the weight network (W_omega)


    """
    def __init__(self, occupancy_function, mesh, weights):
        """

        :param occupancy_function, OccupancyFunction, to use when calculating the occupancy values of sample points
        :param mesh: Trimesh.Mesh, mesh to sample from
        :param weights: Numpy array-like, VxB, vertex weights. (Perhaps should be WeightingFunction)
        """
        self.occupancy_function = occupancy_function
        self.mesh = mesh
        self.weights = weights

        wf = WeightingFunctionPointwise(mesh.vertices, weights)
        self.lbs_deformer = LBSMeshDeformer(mesh.vertices, wf)

    def sample_pose(self, pose, n_bb_points=1024, n_surface_points=1024):
        """

        Create a sample from a pose using a given number of sample points.

        @Additional:
            It may be worth saving surface occupancy and bounding box ocupancy separately.

        :param pose: Numpy array-like, Bx4x4, bone matrices for use in mesh deformation.
        :param n_bb_points: Int, number of points to use within the bounding box of the mesh.
        :param n_surface_points: Int, number of points to use from the surface of the mesh.
        :return: Dict, containing relevant sample data:
                                'vertices': deformed_vertices,
                                'weights': vertex_weights,
                                'occupancy_points': pre-calculated occupancy value for all sample points packed (vertex, occupancy)
        """

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
        result['rest_vertices'] = self.mesh.vertices
        result['pose'] = pose
        result['weights'] = self.weights
        result['occupancy_points'] = occupancy_points

        return result
