import numpy as np
import ex3Utils


# Task 1:
def HoughCircles(imageEdges, radius, votesThresh, distThresh):
    """
    implement the edge version of the circles Hough transform
    :param imageEdges: image represent edge pixels only
    :param radius: array represent the different radiuses to search circles at
    :param votesThresh: threshold represents the minimal number of votes
    required to declare a circle
    :param distThresh: threshold represents the minimal distance between the
    centers of two different circles
    :return: all the circles (center (x,y), radius, votes) detected on the image
    as an Nx4 array where each row is x,y,r,votes.
    """
    yBound, xBound = imageEdges.shape

    # initiate accumulator
    A = np.zeros((xBound, yBound, len(radius)), dtype=int)
    # get only edges from image
    Xs, Ys = np.nonzero(imageEdges)

    radius = np.array(radius)
    thetas = np.arange(0, 360, 1) * np.pi / 180.0
    cosT = np.cos(thetas)
    sinT = np.sin(thetas)
    # voting
    for i in np.arange(0, len(Ys)):
        x, y = Xs[i], Ys[i]
        for r in np.arange(0, len(radius)):
            rad = radius[r]
            # polar coordinates
            a = np.uint(np.round(x - rad * cosT))
            b = np.uint(np.round(y - rad * sinT))
            # get only coordinates in bounds
            goodAs = np.logical_and(0 <= a, a < xBound)
            goodBs = np.logical_and(0 <= b, b < yBound)
            goods = np.argwhere(np.logical_and(goodAs, goodBs)).T
            a = a[goods]
            b = b[goods]
            # vote
            A[(a, b, r)] += 1
    # getting ready for localMaxima
    As, Bs, rs = np.argwhere(A >= votesThresh).T
    votes = A[As, Bs, rs]
    Rs = radius[rs[:]]

    circles = np.vstack((As, Bs, Rs, votes)).T

    localMaxima = ex3Utils.selectLocalMaxima(circles, votesThresh, distThresh)

    return localMaxima


# Task 2:
def bilateralFilter(imgNoisy, spatial_std, range_std):
    """
    WILL FILL LATER
    :param imgNoisy:
    :param spatial_std:
    :param range_std:
    :return:
    """
    return "no"
