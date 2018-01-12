import numpy as np
from scipy.signal import convolve2d

# Task 1: Gaussian pyramid
def gaussianPyramid(img, numOfLevels, filterParam):
    G = {0: img}
    for i in np.arange(1, numOfLevels):
        G[i] = reduce(G[i - 1], filterParam)
    return G

def generate5x5Kernel(filterParam):
    # using gaussian weights
    kernelVector = [0.25 - filterParam / 2.0, 0.25, filterParam, 0.25, 0.25 - filterParam / 2.0]
    kernelVector = np.array(kernelVector)
    return np.outer(kernelVector, kernelVector)

def reduce(img, filterParam):
    # reduce image by convolution with gaussian filter
    kernel = generate5x5Kernel(filterParam)
    # convolution between image and kernel to reduce image by 1/2 (same dim as image)
    convoluted = convolve2d(img, kernel, "same")
    # take second pixel
    return convoluted[::2, ::2]

# bonus
def imConv2(img, kernel1D):
    return 0


# Task 2: Laplacian pyramid
# a
def laplacianPyramid(img, numOfLevels, filterParam):
    return 0


# b
def imgFromLaplacianPyramid(l_pyr, numOfLevels, filterParam):
    return 0


# Task 3: Image blending
def imgBlending(img1, img2, blendingMask, numOfLevels, filterParam):
    return 0
