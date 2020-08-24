
import numpy as np


def extract_voxel_grid(occupancy_function, bounds, initial_resolution, output_file):
    """
    :param occupancy_function:
    :param bounds:
    :param initial_resolution:
    :return:
    """

    bound_dimensions = np.abs(bounds[0] - bounds[1])
    voxel_dimensions = bound_dimensions / initial_resolution
    voxel_grid = np.full(initial_resolution, 0.0)
    points = np.zeros((initial_resolution[0] * initial_resolution[1] * initial_resolution[2], 3))

    for i in range(0, initial_resolution[0]):
        for j in range(0, initial_resolution[1]):
            for k in range(0, initial_resolution[2]):

                index = i * initial_resolution[1] * initial_resolution[2] + j * initial_resolution[2] + k
                voxel_position = bounds[0] + voxel_dimensions * np.array([i + 0.5, j + 0.5, k + 0.5])
                points[index] = voxel_position

    occupancy_map = occupancy_function.evaluate_set(points)
    occupancy_map = occupancy_map.reshape(initial_resolution)
    voxel_grid[occupancy_map > 0.5] = 1

    for i in range(1, initial_resolution[0] - 1):
        for j in range(1, initial_resolution[1] - 1):
            for k in range(1, initial_resolution[2] - 1):

                if (
                    voxel_grid[i + 1][j][k] != 0.0 and voxel_grid[i - 1][j][k] != 0.0 and
                    voxel_grid[i][j + 1][k] != 0.0 and voxel_grid[i][j - 1][k] != 0.0 and
                    voxel_grid[i][j][k + 1] != 0.0 and voxel_grid[i][j][k - 1] != 0.0
                ):

                    voxel_grid[i][j][k] = 0.2

    np.savez(output_file, voxel_start=bounds[0], voxel_dimensions=voxel_dimensions, voxel_grid=voxel_grid)

    return None