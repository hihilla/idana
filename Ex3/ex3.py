import numpy as np
import ex3Utils
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
    :param imgNoisy: image with noise
    :param spatial_std: sigma s
    :param range_std: sigma r
    :return: image that is a result of applying bilateral filter
    """
    M, N = imgNoisy.shape
    imgNoisy = np.asarray(imgNoisy, dtype=float)
    newImg = np.zeros(imgNoisy.shape)
    sigma = int(spatial_std * 3)  # further then that has no influence
    # create kernel to screen image, instead of going over non-influence pixels
    kernel = getKernel(sigma)
    weights = getWeights(kernel, spatial_std)

    for p in np.ndindex(imgNoisy.shape):
        # finding coordinates of surrounding pixels using kernel
        kernelCoordinates = np.int32(repmat([p], kernel.shape[0], 1) + kernel)
        ys, xs = getXsAndYsFrom(kernelCoordinates, M, N)
        # calculate Wpq according to bilateral formula
        Iq = imgNoisy[(ys, xs)]
        Ip = imgNoisy[p]
        tempWs = (Ip - Iq) ** 2
        tempWs = np.exp(-tempWs / (2 * range_std ** 2))
        tempWs = tempWs / tempWs.sum()
        W = weights * tempWs
        # calculate new gray value
        newImg[p] = np.sum(W * Iq) / np.sum(W)
    newImg = np.asarray(newImg, dtype=int)
    return newImg


def getWeights(kernel, spatial_std):
    # each surrounding pixel has a different weight according to it's distance
    tempW = np.abs(np.sum(kernel ** 2, 1))
    tempW = np.exp(-tempW / 2 * (spatial_std ** 2))
    weightsS = tempW / tempW.sum()
    return weightsS


def getKernel(sigma):
    # we want all pixels with coordinates around the current pixel.
    # se we're creating a grid where the top right coordinate is
    # current x - sigma, current y - sigma
    # the bottom left coordinate is current x + sigma, current y + sigma
    # that way we get a kernel-like matrix to know the surrounding coordinates
    # of the current pixel
    Ys, Xs = np.meshgrid(np.linspace(-sigma, sigma, 1 + 2 * sigma),
                         np.linspace(-sigma, sigma, 1 + 2 * sigma))
    kernel = np.vstack((Ys.flatten(), Xs.flatten())).T
    return kernel


def getXsAndYsFrom(kernelCoordinates, yBound, xBound):
    ys = kernelCoordinates[:, 0]
    xs = kernelCoordinates[:, 1]
    ys[ys < 0] = 0
    ys[ys > yBound - 1] = yBound - 1
    xs[xs < 0] = 0
    xs[xs > xBound - 1] = xBound - 1
    return ys, xs
