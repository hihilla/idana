import numpy as np
import scipy.signal.convolve2d as conv

# Task 1: Gaussian pyramid
def gaussianPyramid(img, numOfLevels, filterParam):
    return 0

def generate5x5Kernel(filterParam):
    kernelVector = [0.25 - filterParam / 2.0, 0.25, filterParam, 0.25, 0.25 - filterParam / 2.0]
    kernelVector = np.array(kernelVector)
    np.outer(kernelVector, kernelVector)

def reduce

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
