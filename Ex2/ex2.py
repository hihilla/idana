import numpy as np

# Task 1: Geometrical transformations
def getAffineTransformation(pts1,pts2):
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
    xi = pts1[:,0]
    yi = pts1[:,1]

    #create columns of M
    m0 = np.empty((numOfPoints * 2,), dtype=arrayType)
    m0[0::2] = xi
    m0[1::2] = zeros
    m1 = np.empty((numOfPoints * 2,), dtype = arrayType)
    m1[0::2] = yi
    m1[1::2] = zeros
    m2 = np.empty((numOfPoints * 2,), dtype = arrayType)
    m2[0::2] = zeros
    m2[1::2] = xi
    m3 = np.empty((numOfPoints * 2,), dtype = arrayType)
    m3[0::2] = zeros
    m3[1::2] = yi
    m4 = np.empty((numOfPoints * 2,), dtype = arrayType)
    m4[0::2] = ones
    m4[1::2] = zeros
    m5 = np.empty((numOfPoints * 2,), dtype = arrayType)
    m5[0::2] = zeros
    m5[1::2] = ones
    M = np.column_stack((m0, m1, m2, m3, m4, m5))

    [a,b,c,d,tx,ty] = np.linalg.lstsq(M, bpts2)[0]
    affineT = np.array([[a,b,tx],
                        [c,d,ty],
                        [0,0,1]],
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
    print(xBound, yBound)
    newImg = np.copy(img)
    for x, y in np.ndindex(img.shape):
        newX, newY, _ = np.dot(affineT, [x, y, 1])
        x1 = int(np.floor(newX)) if newX < xBound else xBound
        x2 = int(np.ceil(newX)) if newX < xBound else xBound
        y1 = int(np.floor(newY)) if newY < yBound else yBound
        y2 = int(np.ceil(newY)) if newY < yBound else yBound
        
        pointsWithValue = [[x1, y1, img[x1][y1]],
                           [x1, y2, img[x1][y2]],
                           [x2, y1, img[x2][y1]],
                           [x2, y2, img[x2][y2]]]

        val = bilinearInterpolation(newX, newY, pointsWithValue)
        newImg[x][y] = int(np.around(val)) # new xy ???
    return newImg

def bilinearInterpolation(x, y, pointsWithValue):
    """pointsWithValue are 4 pionts in the form (x, y, value)"""
    pointsWithValue = sorted(pointsWithValue)
    (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = pointsWithValue
    x2 = x2 if x2 < x1 else x1 + 1
    y2 = y2 if y2 < y1 else y1 + 1
    return (q11 * (x2 - x) * (y2 - y) +
            q21 * (x - x1) * (y2 - y) +
            q12 * (x2 - x) * (y - y1) +
            q22 * (x - x1) * (y - y1)
            ) / ((x2 - x1) * (y2 - y1) + 0.0)

def multipleSegmentDefromation(img, Qs, Ps, Qt, Pt, p, b):
    """
    :param: an image and two sets of corresponding lines (draw on two different images).
    Each line is represented by two points (start and end).
    And the two algorithm parameters (p, b)
    :return: the deformed calculated from these lines using the Multiple Segment
    Warping algorithm
    """
    # should use the bilinear interpolation you implemented in 1.b.
    return 0

# Task 2: Image Gradients

def imGradSobel(img):
    #  copies the image to Gx and Gy , padded with double layer of 0's
    newImage = np.copy(img)
    Gx = np.pad(newImage, 2, 'constant', constant_values=0)
    Gy = np.copy(Gx)
    temp = Gx.shape
    x = temp[0]
    y = temp[1]

    horizonalChangeCol = np.array([1, 2, 1])
    horizonalChangeRow = np.array([1, 0, -1])
    changeHorizontal = np.array([[-1, 0, 1],
                                 [-2, 0, 2],
                                 [-1, 0, 1]])

    #  these are just for comfort, can be deleted
    verticalChangeCol = np.array([1, 0, -1])
    verticalChangeRow = np.array([1, 2, 1])
    changeVertical = np.array([[1, 2, 1],
                               [0, 0 , 0],
                              [-1, -2, -1]])

    for i in range(2, x - 2):
        for j in range(2, y - 2):
            toBeChangedGx = np.array([[Gx.item((i-1, j-1)), Gx.item((i, j-1)), Gx.item((i+1, j-1))],
                                   [Gx.item((i-1,j)), Gx.item((i, j)), Gx.item((i+1, j))],
                                   [Gx.item((i-1, j+1)), Gx.item((i, j+1)), Gx.item((i+1, j+1))]])
            toBeChangedGy = np.array([[Gy.item((i-1, j-1)), Gy.item((i, j-1)), Gy.item((i+1, j-1))],
                                   [Gy.item((i-1, j)), Gy.item((i, j)), Gy.item((i+1, j))],
                                   [Gy.item((i-1, j+1)), Gy.item((i, j+1)), Gy.item((i+1, j+1))]])
            temp = np.dot(changeHorizontal, toBeChangedGx)
            Gx[i, j] = np.sum(temp)
            # if i == 100 and j == 100:
            #     print(toBeChangedGx)
            #     print(temp)
            #     print(Gx[i, j])
            # if Gx[i, j] < 0:
            #     Gx[i, j] = 0
            # if Gx[i, j] > 255:
            #     Gx[i, j] = 255
            temp = np.dot(changeVertical, toBeChangedGy)
            Gy[i, j] = temp[1, 1]
            # if Gy[i, j] < 0:
            #     Gy[i, j] = 0
            # if Gy[i, j] > 255:
            #     Gy[i, j] = 255
            #

    toReturn = np.around(np.sqrt(Gx**2 + Gy**2))
    print(toReturn)
    print("%%%%%%%%%%%%%%%%%")
    return Gx, Gy, toReturn