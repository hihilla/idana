import numpy as np
from scipy.signal import convolve2d


# Task 1: Gaussian pyramid
def gaussianPyramid(img, numOfLevels, filterParam):
    G = {0: img}
    for i in np.arange(1, numOfLevels):
        G[i] = reduce(G[i - 1], filterParam)
    return G


def getKernel(filterParam):
    # using gaussian weights
    kernelVector = [0.25 - filterParam / 2.0, 0.25, filterParam, 0.25, 0.25 - filterParam / 2.0]
    kernelVector = np.array(kernelVector)
    return np.outer(kernelVector, kernelVector)


def reduce(img, filterParam):
    # reduce image by convolution with gaussian filter
    kernel = getKernel(filterParam)
    # convolution between image and kernel to reduce image by 1/2 (same dim as image)
    convoluted = imConv2(img, kernel)#convolve2d(img, kernel, "same")
    # take second pixel
    return convoluted[::2, ::2]


# bonus
def imConv2(img, kernel1D):
    Fimg = np.fft.rfft2(img)
    Fkernel = np.fft.rfft2(kernel1D, img.shape)
    # convolution in spatial domain = multiply in frequency domain
    conv = np.fft.irfft2(Fimg * Fkernel).real
    # fix shift
    kernelShape = kernel1D.shape
    conv = np.roll(conv, int(kernelShape[0] / -2.0), 0)
    conv = np.roll(conv, int(kernelShape[0] / -2.0), 1)
    return conv


# Task 2: Laplacian pyramid
# a
def laplacianPyramid(img, numOfLevels, filterParam):
    G = gaussianPyramid(img, numOfLevels, filterParam)
    L = {}
    for i in np.arange(0, numOfLevels - 1):
        Gi = G[i]
        expandedGi1 = expand(G[i + 1], filterParam)
        # make sure they are of the same damnation
        if Gi.shape[0] > expandedGi1.shape[0]:
            # need to remove a row from expandedGi1 to make dimensions equal
            # expandedGi1 = np.delete(expandedGi1, -1, axis=0)
            zeros = np.array([np.zeros(Gi.shape[1], dtype=Gi.dtype)])
            expandedGi1 = np.concatenate((expandedGi1, zeros), axis=0)
        elif Gi.shape[1] > expandedGi1.shape[1]:
            # need to remove a column from expandedGi1 to make dimensions equal
            # expandedGi1 = np.delete(expandedGi1, -1, axis=1)
            zeros = np.array([np.zeros(Gi.shape[0], dtype=Gi.dtype)])
            expandedGi1 = np.concatenate((expandedGi1, zeros), axis=1)
        L[i] = Gi - expandedGi1
    L[numOfLevels - 1] = G[numOfLevels - 1]
    return L


def expand(img, filterParam):
    # expand image by convolution with gaussian filter
    kernel = getKernel(filterParam)
    # create an image twice as big, each second pixel is from original image,
    # using convolution with gaussian
    newShape = (img.shape[0] * 2, img.shape[1] * 2)
    expandedImg = np.zeros(newShape, dtype=img.dtype)
    expandedImg[::2, ::2] = img[:, :]
    return 4 * convolve2d(expandedImg, kernel, "same")


# b
def imgFromLaplacianPyramid(laplacPrmd, numOfLevels, filterParam):
    # take L[i], expand it, add to L[i - 1]
    outImg = np.zeros(laplacPrmd[0].shape, dtype=laplacPrmd[0].dtype)
    for i in np.arange(numOfLevels - 1, 0, -1):
        Li = expand(laplacPrmd[i], filterParam)
        Li1 = laplacPrmd[i - 1]
        # make sure they are of the same damnation
        if Li.shape[0] > Li1.shape[0]:
            zeros = np.array([np.zeros(Li.shape[1], dtype=Li.dtype)])
            Li1 = np.concatenate((Li1, zeros), axis=0)
            # Li = np.delete(Li, -1, axis=0)
        elif Li.shape[1] > Li1.shape[1]:
            zeros = np.array([np.zeros(Li.shape[0], dtype=Li.dtype)])
            Li1 = np.concatenate((Li1, zeros), axis=1)
            # Li = np.delete(Li, -1, axis=1)
        outImg = Li + Li1
        laplacPrmd[i - 1] = outImg
    return outImg


# Task 3: Image blending
def imgBlending(img1, img2, blendingMask, numOfLevels, filterParam):
    return 0
