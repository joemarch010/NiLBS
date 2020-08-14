"""

Implementation of the generalised winding numbers algorithm.

Based on work from: https://github.com/marmakoide/inside-3d-mesh

@Note
    This could be re-implemented in TensorFlow if it becomes too slow

"""

import numpy as np

'''
Naive and straightforward implementation of the inside/outside point mesh test
'''


def is_inside_naive(triangles, X):
    # Compute triangle vertices and their norms relative to X
    M = triangles - X
    M_norm = np.sqrt(np.sum(M ** 2, axis=2))

    # Accumulate generalized winding number per triangle
    winding_number = 0.
    for (A, B, C), (a, b, c) in zip(M, M_norm):
        winding_number += np.arctan2(np.linalg.det(np.array([A, B, C])),
                                        (a * b * c) + c * np.dot(A, B) + a * np.dot(B, C) + b * np.dot(C, A))

    # Job done
    return winding_number >= 2. * np.pi


'''
Optimized for numpy implementation of the inside/outside point mesh test
'''


def is_inside_turbo(triangles, X):
    # Compute euclidean norm along axis 1
    def anorm2(X):
        return np.sqrt(np.sum(X ** 2, axis=1))

    # Compute 3x3 determinant along axis 1
    def adet(X, Y, Z):
        ret = np.multiply(np.multiply(X[:, 0], Y[:, 1]), Z[:, 2])
        ret += np.multiply(np.multiply(Y[:, 0], Z[:, 1]), X[:, 2])
        ret += np.multiply(np.multiply(Z[:, 0], X[:, 1]), Y[:, 2])
        ret -= np.multiply(np.multiply(Z[:, 0], Y[:, 1]), X[:, 2])
        ret -= np.multiply(np.multiply(Y[:, 0], X[:, 1]), Z[:, 2])
        ret -= np.multiply(np.multiply(X[:, 0], Z[:, 1]), Y[:, 2])
        return ret

    # One generalized winding number per input vertex
    ret = np.zeros(X.shape[0], dtype=X.dtype)

    # Accumulate generalized winding number for each triangle
    for U, V, W in triangles:
        A, B, C = U - X, V - X, W - X
        omega = adet(A, B, C)

        a, b, c = anorm2(A), anorm2(B), anorm2(C)
        k = a * b * c
        k += c * np.sum(np.multiply(A, B), axis=1)
        k += a * np.sum(np.multiply(B, C), axis=1)
        k += b * np.sum(np.multiply(C, A), axis=1)

        ret += np.arctan2(omega, k)

    # Job done
    return ret >= 2 * np.pi