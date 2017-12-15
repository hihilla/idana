import numpy as np
import ex3Utils
import numpy.linalg as la
from numpy.matlib import repmat


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
    Implement Bilateral Filter
    :param imgNoisy: noist image
    :param spatial_std: sigma s
    :param range_std: sigma r
    :return: image that is a result of applying bilateral filter
    """
    M, N = imgNoisy.shape
    newImg = np.zeros(imgNoisy.shape)
    sigma = int(spatial_std * 3)  # further then that has no influence
    # create kernel to screen image, instead of going over non-influence pixels
    kernel = getKernel(sigma)
    weights = getWheights(kernel, spatial_std)

    for p in np.ndindex(imgNoisy.shape):
        kernelCoordinates = np.uint32(repmat([p], kernel.shape[0], 1) + kernel)
        ys, xs = getXsAndYs(M, N, kernelCoordinates)
        # calculate Wpq according to bilateral formula
        pixelsAround = imgNoisy[(ys, xs)]
        tempWs = (pixelsAround - imgNoisy[p]) ** 2
        tempWs = np.exp(-tempWs / (2 * range_std ** 2))
        tempWs = tempWs / tempWs.sum()
        W = weights * tempWs
        # calculate new gray value
        newImg[p] = np.sum(pixelsAround * W) / np.sum(W)
    newImg = np.asarray(newImg, dtype=int)
    return newImg


def getWheights(kernel, spatial_std):
    tempW = np.abs(np.sum(kernel ** 2, 1))
    tempW = np.exp(-tempW / 2 * (spatial_std ** 2))
    weightsS = tempW / tempW.sum()
    return weightsS


def getKernel(sigma):
    Ys, Xs = np.meshgrid(np.linspace(-sigma, sigma, 1 + 2 * sigma),
                         np.linspace(-sigma, sigma, 1 + 2 * sigma))
    kernel = np.vstack((Ys.flatten(), Xs.flatten())).T
    return kernel


def getXsAndYs(yBound, xBound, kernelCoordinates):
    ys = kernelCoordinates[:, 0]
    xs = kernelCoordinates[:, 1]
    ys[ys < 0] = 0
    ys[ys > yBound - 1] = yBound - 1
    xs[xs < 0] = 0
    xs[xs > xBound - 1] = xBound - 1
    return ys, xs

