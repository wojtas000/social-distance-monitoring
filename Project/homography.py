import torch
import numpy as np

"""
This module contains functions calculating and applying the homography. 

"""


def compute_homography(points1: tuple, points2: tuple) -> np.array:

    """
    Calculate homography matrix between the camera views

    :param points1: <tuple> corner points of image from first camera view 
    :param points2: <tuple> corner points of image from second camera view (we treat this as birds-eye view)
    :return: <np.array> homography matrix between the views

    """
    n = len(points1)
    array_list = [0] * n
    for i in range(n):
        x, y = points1[i]
        u, v = points2[i]
        array_list[i] = np.array([
                                [-x, -y, -1, 0, 0, 0, u*x, u*y, u],
                                [0, 0, 0, -x, -y, -1, v*x, v*y, v]
                                ])
    matrix = np.concatenate(array_list, axis=0)

    # from SVD decomposition of A we get V transposed.
    # Last column of V is vector, which will form homography matrix

    _, _, v_transposed = np.linalg.svd(matrix)
    h = v_transposed[-1, :]

    # We want the last element of matrix to be 1, so we scale the matrix.

    scaling_factor = h[-1]
    homography = h.reshape((3, 3))/scaling_factor

    return homography


def homography_on_point(point: np.array, homography: np.array) -> np.array:
    """
    Calculate coordinates of point after applying homography

    :param point: <numpy.array> point to be transformed
    :param homography: <numpy.array> homography matrix
    :return: <numpy.array> transformed point (coordinates rounded to the nearest integer values)
    """

    x = homography @ np.array([point[0], point[1], 1])
    x = x/x[2]
    x = np.round(x).astype(int)
    return np.array([x[0], x[1]])


def apply_homography_to_image(img: torch.tensor, homography: np.array) -> torch.tensor:
    """
    Apply homography transformation to given image

    :param img: <torch.tensor> image we want to transform
    :param homography: <numpy.array> homography matrix, used for transformation
    :return: <np.array> transformed image
    """
    height = img.shape[1]
    width = img.shape[2]
    warped_image = torch.zeros(img.shape)

    # assigning pixel RGB values to right places in the transformed image

    for i in range(height):
        for j in range(width):
            coord = homography_on_point(np.array([j, i]), homography)
            if (coord[0] < 0 or coord[0] >= width) or (coord[1] < 0 or coord[1] >= height):
                continue
            else:
                warped_image[:, coord[1], coord[0]] = img[:, i, j]

    return warped_image
