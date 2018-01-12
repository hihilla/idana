import numpy as np
from scipy.signal import convolve2d

# Task 1: Gaussian pyramid
def gaussianPyramid(img, numOfLevels, filterParam=0.4):
    G = {0: img}

    for i in range(1, numOfLevels):
        # every entry of dict is the previous after reduce function
        newImage = reduce(G[i - 1], filterParam)
        G[i] = newImage

    return G


def genreateKernel(a):
    weightsOneDim = np.array([0.25 - a / 2.0, 0.25, a , 0.25, 0.25 - a / 2.0])
    weights = np.outer(weightsOneDim, weightsOneDim)

    return weights


def reduce(image, filterParam):
    kernel = genreateKernel(filterParam)
    newImage = convolve2d(image, kernel, 'same')
    # returns only every second pixel of the image after convelution
    return newImage[::2, ::2]


# bonus
def imConv2(img, kernel1D):
    return 0


# Task 2: Laplacian pyramid
# a
def laplacianPyramid(img, numOfLevels, filterParam=0.4):
    L = {}
    G = gaussianPyramid(img, numOfLevels, filterParam)

    for i in range(0, numOfLevels - 1):
        Gi = G[i]
        expandedG = expand(G[i + 1], filterParam)

        # fixes the size of the unexpanded level if needed
        # by duplicating the last row/col
        if expandedG.shape[0] > Gi.shape[0]:
            newRow = np.array(Gi[Gi.shape[0] - 1])
            Gi = np.concatenate(Gi, newRow)
        if expandedG.shape[1] > Gi.shape[1]:
            newCol = np.array(Gi[:, 0])
            Gi = np.concatenate(Gi, newCol)
        L[i] = Gi - expandedG

    L[numOfLevels - 1] = G[numOfLevels - 1]

    return L


def expand(image, filterParam):
    kernel = genreateKernel(filterParam)
    # creates an image twice the size,
    # only fills every other pixel with values of given image
    # the rest are zeros
    # convulotion is on the half zeros-half given image
    newImage = np.zeros((image.shape[0] * 2, image.shape[1] * 2))
    newImage[::2, ::2] = image[:, :]
    # multiply by 4 to normalize after kernel (divides by 4)
    newImage = 4 * convolve2d(newImage, kernel, 'same')

    return newImage


# b
def imgFromLaplacianPyramid(l_pyr, numOfLevels, filterParam=0.4):
    # creates the size of the final image to be returned
    image = np.zeros(l_pyr[0].shape[0], l_pyr[1].shape[1])

    for i in range(numOfLevels - 1, 0, -1):
        laplacExpanded = expand(l_pyr[i])
        reconstructedLevel = l_pyr[i - 1] + laplacExpanded

    return 0


# Task 3: Image blending
def imgBlending(img1, img2, blendingMask, numOfLevels, filterParam=0.4):
    return 0
