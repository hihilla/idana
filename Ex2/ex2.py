import numpy as np


# Task 1: Geometrical transformations
def getAffineTransformation(pts1, pts2):
    """
    :param: pts1,pts2 - at least 3 pairs of matched points between images A and B
    :return: an affine transformation from image A to image B
    """
    # built-in numpy method for finding the least-squares solution for
    # linear systems using Singular Value Decomposition (SVD)
    # https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.lstsq.html
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
    # should implement a bi-linear interpolation function to calculate new pixel values
    xBound, yBound = img.shape
    xBound -= 1
    yBound -= 1
    newImg = np.zeros(img.shape)
    for y, x in np.ndindex(img.shape):
        newx, newy, _ = np.dot(affineT, [x, y, 1])
        val = bilinearInterpolation(img, x, y)
        newx = np.around(np.clip(newx, 0, xBound)).astype(int)
        newy = np.around(np.clip(newy, 0, yBound)).astype(int)
        newImg[int(newy)][int(newx)] = val
    return newImg


def bilinearInterpolation(img, x, y):
    yBound, xBound = img.shape
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, xBound - 1)
    x1 = np.clip(x1, 0, xBound - 1)
    y0 = np.clip(y0, 0, yBound - 1)
    y1 = np.clip(y1, 0, yBound - 1)

    q00 = img[y0, x0]
    q10 = img[y1, x0]
    q01 = img[y0, x1]
    q11 = img[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return int(wa * q00 + wb * q10 + wc * q01 + wd * q11)


def singleSegmentationDeformation(Rt, Qt, Pt, Qs, Ps):
    normQPt = np.linalg.norm(Qt - Pt)
    ut = (Qt - Pt) / normQPt
    vt = np.array([ut[1], -ut[0]])

    alpha = np.dot((Rt - Pt), ut) / normQPt
    beita = np.dot((Rt - Pt), vt)

    normQPs = np.linalg.norm(Qs - Ps)
    u = (Qs - Ps) / normQPs
    v = np.array([u[1], -u[0]])

    R = Ps + np.dot(np.dot(alpha, normQPs), u) + np.dot(beita, v)

    return R, beita


def multipleSegmentDefromation(img, Qs, Ps, Qt, Pt, p, b):
    a = 0.001
    newImg = np.zeros(img.shape)
    for y, x in np.ndindex(img.shape):
        Rt = np.array([x, y])
        sum1 = 0
        sum2 = 0
        for i in range(0, len(Qs)):
            Ri, beita = singleSegmentationDeformation(Rt, Qt[i], Pt[i],
                                                      Qs[i], Ps[i])
            temp = np.abs(Qs[i] - Ps[i])
            temp = np.power(temp, p)
            temp /= (a + beita)
            wi = np.power(temp, b)
            sum1 += (wi * Ri)
            sum2 += wi

        R = sum1 / sum2
        R = np.asarray(R, int)
        val = bilinearInterpolation(img, x, y)
        newImg[R[1], R[0]] = val
    return newImg


# Task 2: Image Gradients

def imGradSobel(img):
    newImage = np.copy(img)

    #  both Gx,Gy matrices are same size to make to sum of them easier later
    #  pads each with reflected values, one row\col of each direction
    newImage = np.lib.pad(newImage, ((1, 1),), 'reflect')
    Gx = newImage
    Gy = newImage

    # kernels are already reflected
    kernelRow = np.array([1, 0, -1])
    kernelCol = np.array([1, 2, 1])
    rows = Gx.shape[0]
    cols = Gx.shape[1]

    # each 3-tuple that at least the mid element is in the original
    # image are being multiplied with the mask, the result
    # replace the mid element of the tuple in Gx matrix
    for i in range(1, rows - 1):
        for j in range(0, cols):
            currVector = np.array([newImage[i - 1, j], newImage[i, j], newImage[i + 1, j]])
            newVal = np.dot(currVector, kernelRow)
            # if newVal < 0:
            #     newVal = 0
            # elif newVal > 255:
            #     newVal = 255
            Gx[i, j] = newVal
    # same for cols in Gy matrix
    for j in range(1, cols - 1):
        for i in range(0, rows):
            currVector = np.array([newImage[i, j - 1], newImage[i, j], newImage[i, j + 1]])
            newVal = np.dot(currVector, kernelCol)
            # if newVal < 0:
            #     newVal = 0
            # elif newVal > 255:
            #     newVal = 255
            Gy[i, j] = newVal

    Gx = Gx[1:rows - 1, 1:cols - 1]
    Gy = Gy[1:rows - 1, 1:cols - 1]

    Gx = np.array(Gx, dtype=int)
    Gy = np.array(Gy, dtype=int)
    temp = np.power(Gx, 2) + np.power(Gy, 2)
    magnitude = np.asarray(np.sqrt(temp), int)

    return Gx, Gy, magnitude
