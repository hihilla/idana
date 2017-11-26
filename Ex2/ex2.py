import numpy as np


# Task 1: Geometrical transformations
def getAffineTransformation(pts1, pts2):
    """
    :param: pts1,pts2 - at least 3 pairs of matched points between images A and B
    :return: an affine transformation from image A to image B
    """
    bpts2 = np.hstack(pts2)
    # create arrays xi, yi, zeros, ones
    arrayType = pts1.dtype
    numOfPoints = int(pts1.size / 2.0)
    zeros = np.zeros((numOfPoints,), dtype=int)
    ones = np.ones((numOfPoints,), dtype=int)
    xi = pts1[:, 0]
    yi = pts1[:, 1]

    # create columns of M
    m0 = np.empty((numOfPoints * 2,), dtype=arrayType)
    m0[0::2] = xi
    m0[1::2] = zeros
    m1 = np.empty((numOfPoints * 2,), dtype=arrayType)
    m1[0::2] = yi
    m1[1::2] = zeros
    m2 = np.empty((numOfPoints * 2,), dtype=arrayType)
    m2[0::2] = zeros
    m2[1::2] = xi
    m3 = np.empty((numOfPoints * 2,), dtype=arrayType)
    m3[0::2] = zeros
    m3[1::2] = yi
    m4 = np.empty((numOfPoints * 2,), dtype=arrayType)
    m4[0::2] = ones
    m4[1::2] = zeros
    m5 = np.empty((numOfPoints * 2,), dtype=arrayType)
    m5[0::2] = zeros
    m5[1::2] = ones
    M = np.column_stack((m0, m1, m2, m3, m4, m5))

    [a, b, c, d, tx, ty] = np.linalg.lstsq(M, bpts2)[0]
    affineT = np.array([[a, b, tx],
                        [c, d, ty],
                        [0, 0, 1]],
                       dtype=float)

    return np.array(affineT, dtype=float)


def applyAffineTransToImage(img, affineT):
    """
    :param: an image A and an affine transformation T
    :return: the transformed image T*A
    """
    # implement a bi-linear interpolation function to calculate new pixel values
    xBound, yBound = img.shape
    xBound -= 1
    yBound -= 1
    newImg = np.zeros(img.shape)
    for y, x in np.ndindex(img.shape):
        newx, newy, _ = np.dot(affineT, [x, y, 1])
        # val = bilinear_interpolate(img, x, y)
        val = bilinearInterpolation(x, y, img)
        newx = np.around(np.clip(newx, 0, xBound)).astype(int)
        newy = np.around(np.clip(newy, 0, yBound)).astype(int)
        newImg[int(newy)][int(newx)] = val
    return newImg

#
# def bilinear_interpolate(im, x, y):
#     x0 = np.floor(x).astype(int)
#     x1 = x0 + 1
#     y0 = np.floor(y).astype(int)
#     y1 = y0 + 1
#
#     x0 = np.clip(x0, 0, im.shape[1] - 1)
#     x1 = np.clip(x1, 0, im.shape[1] - 1)
#     y0 = np.clip(y0, 0, im.shape[0] - 1)
#     y1 = np.clip(y1, 0, im.shape[0] - 1)
#
#     Ia = im[y0, x0]
#     Ib = im[y1, x0]
#     Ic = im[y0, x1]
#     Id = im[y1, x1]
#
#     wa = (x1 - x) * (y1 - y)
#     wb = (x1 - x) * (y - y0)
#     wc = (x - x0) * (y1 - y)
#     wd = (x - x0) * (y - y0)
#
#     return int(wa * Ia + wb * Ib + wc * Ic + wd * Id)


def bilinearInterpolation(x, y, img):
    xBound, yBound = img.shape
    x1 = np.floor(x).astype(int)
    x2 = x1 + 1
    y1 = np.floor(y).astype(int)
    y2 = y1 + 1

    x1 = np.clip(x1, 0, xBound - 1)
    x2 = np.clip(x2, 0, xBound - 1)
    y1 = np.clip(y1, 0, yBound - 1)
    y2 = np.clip(y2, 0, yBound - 1)

    q11 = img[y1][x1] if x < xBound and y < yBound else 0
    q21 = img[y1][x2] if x < xBound and y < yBound else 0
    q12 = img[y2][x1] if x < xBound and y < yBound else 0
    q22 = img[y2][x2] if x < xBound and y < yBound else 0

    A = np.array([[1, x1, y1, x1 * y1],
                  [1, x1, y2, x1 * y2],
                  [1, x2, y1, x2 * y1],
                  [1, x2, y2, x2 * y2]])
    Q = np.array([[q11, q12, q21, q22]])

    coeff = np.linalg.lstsq(A, Q.T)[0]

    val = coeff[0][0] + coeff[1][0] * x + coeff[2][0] * y + coeff[3][0] * x * y

    return int(val)


def multipleSegmentDefromation(img, Qs, Ps, Qt, Pt, p, b):
    """
    :param: an image and two sets of corresponding lines (draw on two different images).
    Each line is represented by two points (start and end).
    And the two algorithm parameters (p, b)
    :return: the deformed calculated from these lines using the Multiple Segment
    Warping algorithm
    """
    # use the bi-linear interpolation implemented in 1.b.
    P = [Ps, Pt]
    Q = [Qs, Qt]
    # The influence of each segments
    W = np.array((2,), dtype=float)
    for y, x, _ in np.ndindex(img.shape):
        R = (x, y)

    return 0


# Task 2: Image Gradients

def imGradSobel(img):
    return 0
