import numpy as np
from scipy.spatial import Delaunay


"""
  Helper Function - Do Not Modify
  You can use this function in your code
"""


def matrixABC(sparse_control_points, elements):
    """
    Get the triangle matrix given three endpoint sets
    [[ax bx cx]
     [ay by cy]
     [1   1  1]]

    Input -
    sparse_control_points - sparse control points for the input image
    elements - elements (Each Simplex) of Tri.simplices

    Output -
    Stack of all [[ax bx cx]
                  [ay by cy]
                  [1   1  1]]
    """
    output = np.zeros((3, 3))

    # First two rows using Ax Ay Bx By Cx Cy
    for i, element in enumerate(elements):
        output[0:2, i] = sparse_control_points[element, :]

    # Fill last row with 1s
    output[2, :] = 1

    return output


"""
	Helper Function - Do Not Modify
	You can use this helper function in generate_warp
"""


def interp2(v, xq, yq):
    dim_input = 1
    if len(xq.shape) == 2 or len(yq.shape) == 2:
        dim_input = 2
        q_h = xq.shape[0]
        q_w = xq.shape[1]
        xq = xq.flatten()
        yq = yq.flatten()

    h = v.shape[0]
    w = v.shape[1]
    if xq.shape != yq.shape:
        raise ("query coordinates Xq Yq should have same shape")

    x_floor = np.floor(xq).astype(np.int32)
    y_floor = np.floor(yq).astype(np.int32)
    x_ceil = np.ceil(xq).astype(np.int32)
    y_ceil = np.ceil(yq).astype(np.int32)

    x_floor[x_floor < 0] = 0
    y_floor[y_floor < 0] = 0
    x_ceil[x_ceil < 0] = 0
    y_ceil[y_ceil < 0] = 0

    x_floor[x_floor >= w - 1] = w - 1
    y_floor[y_floor >= h - 1] = h - 1
    x_ceil[x_ceil >= w - 1] = w - 1
    y_ceil[y_ceil >= h - 1] = h - 1

    v1 = v[y_floor, x_floor]
    v2 = v[y_floor, x_ceil]
    v3 = v[y_ceil, x_floor]
    v4 = v[y_ceil, x_ceil]

    lh = yq - y_floor
    lw = xq - x_floor
    hh = 1 - lh
    hw = 1 - lw

    w1 = hh * hw
    w2 = hh * lw
    w3 = lh * hw
    w4 = lh * lw

    interp_val = v1 * w1 + w2 * v2 + w3 * v3 + w4 * v4

    if dim_input == 2:
        return interp_val.reshape(q_h, q_w)
    return interp_val


"""
Function - Modify
"""


def generate_warp(size_H, size_W, Tri, ABC_Inter_inv_set, ABC_im_set, image):
    """
    INPUT
      Generates warping of input image and returns the warped image
      size_H - Height of Image
      size_W -  Width of Image
      Tri - Delanauy Triangulations (generated from scipy.spatial library)
      ABC_Inter_inv_set - Stack of all Inverted ABC Matrices
      ABC_im_set - Stack of all ABC matrix triangles for the input image
      image - input image - size_H x size_W x 3
    OUPUT
      generated_pic - size_H x size_W x 3 Warped image
    """
    # TODO: Implement here
    # Generate x,y meshgrid
    grid = np.meshgrid(np.arange(size_W), np.arange(size_H))
    x, y = grid[0], grid[1]

    # Flatten the meshgrid
    x = x.flatten()
    y = y.flatten()

    points = np.vstack((x, y)).T
    simplices = Tri.find_simplex( points )  # simplices[i] is the index of the triangle which contains the i-th point

    # Compute alpha, beta, gamma for all the color layers(3)
    alpha = np.zeros((size_H * size_W))
    beta = np.zeros((size_H * size_W))
    gamma = np.zeros((size_H * size_W))

    ABC_inv_per_point = ABC_Inter_inv_set[
        simplices
    ]  # ABC_inv_per_point[i] is the inverted ABC matrix of the triangle which contains the i-th point
    alpha = (
        ABC_inv_per_point[:, 0, 0] * points[:, 0]
        + ABC_inv_per_point[:, 0, 1] * points[:, 1]
        + ABC_inv_per_point[:, 0, 2] * 1
    )
    beta = (
        ABC_inv_per_point[:, 1, 0] * points[:, 0]
        + ABC_inv_per_point[:, 1, 1] * points[:, 1]
        + ABC_inv_per_point[:, 1, 2] * 1
    )
    gamma = (
        ABC_inv_per_point[:, 2, 0] * points[:, 0]
        + ABC_inv_per_point[:, 2, 1] * points[:, 1]
        + ABC_inv_per_point[:, 2, 2] * 1
    )

    # Find all x and y coordinates
    source_x = (
        alpha * ABC_im_set[simplices, 0, 0]
        + beta * ABC_im_set[simplices, 0, 1]
        + gamma * ABC_im_set[simplices, 0, 2]
    )
    source_y = (
        alpha * ABC_im_set[simplices, 1, 0]
        + beta * ABC_im_set[simplices, 1, 1]
        + gamma * ABC_im_set[simplices, 1, 2]
    )

    # Generate Warped Images (Use function interp2) for each of 3 layers
    generated_pic = np.zeros((size_H, size_W, image.shape[-1]), dtype=np.uint8)
    source_points = np.vstack((source_x, source_y)).T.astype(int)
    generated_pic[:, :, :] = image[:, :, :][source_points[:, 1], source_points[:, 0]].reshape(generated_pic.shape)

    return generated_pic


"""
Function - Do Not Modify
"""


def ImageMorphingTriangulation(
    im1,
    im2,
    im1_pts,
    im2_pts,
    warp_frac,
    dissolve_frac,
    use_dst_points_for_triangles=True,
):
    """
    INPUT
      im1: H×W×3 numpy array representing the first image.
      im2: H×W×3 matrix representing the second image.
      im1_pts: N×2 matrix representing correspondences in the first image.
      im2_pts: N×2 matrix representing correspondences in the second image.
      warp_frac: scalar representing each frame’s shape warping parameter.
      dissolve_frac: scalar representing each frame’s cross-dissolve parameter.

    OUTPUT
      H×W×3 numpy array representing the morphed image

    Tips: Read about Delaunay function from scipy.spatial and see how you could
    use it here
    """

    # compute the H,W of the images (same size)
    size_H = im1.shape[0]
    size_W = im1.shape[1]

    if use_dst_points_for_triangles:
        img_coor_inter = im2_pts
    else:
        # find the averagen of the sparse control points of the two images
        img_coor_inter = (im1_pts + im2_pts) / 2

    # create a new triangulation of the intermediate points
    Tri = Delaunay(img_coor_inter)
    
    # No. of Triangles
    nTri = Tri.simplices.shape[0]

    # Initialize the Triangle Matrices for all the triangles in image
    ABC_Inter_inv_set = np.zeros((nTri, 3, 3))
    ABC_im1_set = np.zeros((nTri, 3, 3))
    ABC_im2_set = np.zeros((nTri, 3, 3))

    for ii, element in enumerate(Tri.simplices):
        ABC_Inter_inv_set[ii, :, :] = np.linalg.inv(
            matrixABC(img_coor_inter, element)
        )
        ABC_im1_set[ii, :, :] = matrixABC(im1_pts, element)
        ABC_im2_set[ii, :, :] = matrixABC(im2_pts, element)

    assert ABC_Inter_inv_set.shape[0] == nTri

    # generate warp pictures for each of the two images
    if dissolve_frac != 1:
        warp_im1 = generate_warp(
            size_H, size_W, Tri, ABC_Inter_inv_set, ABC_im1_set, im1
        )
    else:
        warp_im1 = im1
    if dissolve_frac != 0:
        warp_im2 = generate_warp(
            size_H, size_W, Tri, ABC_Inter_inv_set, ABC_im2_set, im2
        )
    else:
        warp_im2 = im2

    # dissolve process
    dissolved_pic = (1 - dissolve_frac) * warp_im1 + dissolve_frac * warp_im2
    dissolved_pic = dissolved_pic.astype(np.uint8)

    return dissolved_pic