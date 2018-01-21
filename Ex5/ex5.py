import numpy as np
from scipy.signal import convolve2d


# Task 1: Gaussian pyramid
def gaussianPyramid(img, numOfLevels, filterParam):
    G = {0: img}
    for i in range(1, numOfLevels):
        # every entry of dict is the previous after reduce function
        G[i] = reduce(G[i - 1], filterParam)
    return G


def getKernel(filterParam):
    # using gaussian weights
    return np.array([0.25 - filterParam / 2.0, 0.25, filterParam, 0.25, 0.25 - filterParam / 2.0])


def getKernel2d(filterParam):
    vector = getKernel(filterParam)
    return np.outer(vector, vector)


def reduce(image, filterParam):
    kernel = getKernel(filterParam)
    # taking only every second pixel of the image after convolution
    newImage = imConv2(image, kernel)
    newImage = np.array(newImage[::2, ::2])
    return newImage


# bonus
def imConv2(img, kernel1D):
    padding = int(kernel1D.shape[0] / 2.0)
    # adding padding to avoid overflow
    imgPad = np.pad(img, padding, 'constant')

    tempResX = np.zeros(imgPad.shape)
    tempResY = np.zeros(imgPad.shape)

    # separate convolution to rows and columns
    for i in np.arange(0, kernel1D.shape[0]):
        window = imgPad[:, i:imgPad.shape[0] - 2 * padding + i]
        tempResY[:, padding:-padding] = tempResY[:, padding:-padding] + window * kernel1D[i]

    for i in np.arange(0, kernel1D.shape[0]):
        window = tempResY[i:imgPad.shape[1] - 2 * padding + i, :]
        tempResX[padding:-padding, :] = tempResX[padding:-padding, :] + window * kernel1D[i]

    # ignoring zero padding
    return tempResX[padding:img.shape[0] + padding, padding:img.shape[1] + padding]


# Task 2: Laplacian pyramid
# a
def laplacianPyramid(img, numOfLevels, filterParam):
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
    kernel = getKernel(filterParam)
    # creates an image twice the size,
    # only fills every other pixel with values of given image
    # the rest are zeros
    newImage = np.zeros((image.shape[0] * 2, image.shape[1] * 2))
    newImage[::2, ::2] = image[:, :]
    # multiply by 4 to normalize after kernel (divides by 4)
    newImage = 4 * imConv2(newImage, kernel)
    return np.array(newImage, dtype=int)


# b
def imgFromLaplacianPyramid(laplacePrmd, numOfLevels, filterParam):
    # creates the size of the final image to be returned
    image = np.zeros(laplacePrmd[0].shape)
    for i in range(numOfLevels - 1, 0, -1):
        eLi = expand(laplacePrmd[i], filterParam)
        Li1 = laplacePrmd[i - 1]
        # fixes the expanded to be as the size of the previous
        # level of the pyramid for future valid summary
        if eLi.shape[0] > Li1.shape[0]:
            eLi = np.delete(eLi, 0, axis=0)
        if eLi.shape[1] > Li1.shape[1]:
            eLi = np.delete(eLi, 0, axis=1)
        reconstructedLevel = Li1 + eLi
        laplacePrmd[i - 1] = reconstructedLevel
        image = reconstructedLevel
    return image


# Task 3: Image blending
def imgBlending(img1, img2, blendingMask, numOfLevels, filterParam):
    # Build Laplacian pyramids LA and LB from images A and B
    LA = laplacianPyramid(img1, numOfLevels, filterParam)
    LB = laplacianPyramid(img2, numOfLevels, filterParam)
    # Build a Gaussian pyramid GM from selected mask
    GM = gaussianPyramid(blendingMask, numOfLevels, filterParam)
    # Form a combined pyramid LS from LA and LB using nodes of GM as weights
    LS = {}
    for key in LA.keys():
        LS[key] = GM[key] * LA[key] + (1 - GM[key]) * LB[key]
    # Collapse the LS pyramid to get the final blended image
    blendImg = imgFromLaplacianPyramid(LS, numOfLevels, filterParam)
    return blendImg
