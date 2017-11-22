import numpy as np

# Task 1: Spatial sampling
def getSampledImageAtResolution(dim, pixelSize, k=2):
    x0 = dim[0]
    x1 = dim[1]
    y0 = dim[2]
    y1 = dim[3]

    numXPixels = int(float(x1 - x0) / pixelSize)
    numYPixels = int(float(y1 - y0) / pixelSize)

    X = np.linspace(dim[0], dim[1], numXPixels)
    Y = np.linspace(dim[0], dim[1], numYPixels)

    Xs, Ys = np.meshgrid(X, Y)
    Xs = np.multiply(3.0, Xs)
    Ys = np.multiply(2.0, Ys)
    return np.cos(k * np.pi * (Xs + Ys))
# HILLAS CODE
def optimalQuantizationImage(img, k):
    # numericalVal = img[:, :, 0]
    imgSize = np.size(img) / 3.0

    # initial k random centroids
    centroids = np.random.randint(0, 256, k)
    centroids = np.sort(centroids)

    # compute the boundary values
    bounds = np.zeros(k + 1, float)
    bounds[0] = 0
    bounds[k] = 255
    for i in range(1, k):
        bounds[i] = (centroids[i - 1] + centroids[i]) / 2.0

    P = calcProbs(imgSize, getImageHistogram(img))

    centroidsChanged = False
    while centroidsChanged:
        prevCentroids = np.copy(centroids)
        for i in range(0, k):
            zi = int(bounds[i])
            zi1 = int(bounds[i + 1])
            sum1 = 0
            sum2 = 0
            for z in range(zi, zi1 + 1):
                sum1 += (z * P[z])
                sum2 += P[z]
            centroids[i] = int(sum1 / sum2 if sum2 != 0 else 0)
            if i != 0:
                bounds[i] = (centroids[i - 1] + centroids[i]) / 2.0
        centroidsChanged = not (np.array_equiv(centroids, prevCentroids))
    print(centroids)
    newImage = np.copy(img)
    for x, y, _ in np.ndindex(newImage.shape):
        for i in range(0, len(bounds) - 1):
            color = newImage[x][y][0]
            if bounds[i] <= color <= bounds[i + 1]:
                newImage[x][y] = np.repeat(centroids[i], 3)
                break
    return newImage

# ADARS CODE
# Task 2: Quantization
# def optimalQuantizationImage(img, k):
#     numericalVals = img[:,:,0]
#     numOfPixels = np.size(img) / 3.0
#
#     epsilon = 0.001
#
#     # choosing centroids
#     centroids = np.random.randint(0, 256, k)
#     centroids = np.sort(centroids)
#
#     # bounds the centroids: the middle between two adjacent centroids
#     bounds = np.zeros(k + 1, float)
#     bounds[0] = 0
#     bounds[k] = 255
#     bounds = calcBounds(k, bounds, centroids)
#     probs = calcProbs(numOfPixels, getImageHistogram(img))
#     prevErr = calcErr(centroids, bounds, numericalVals, probs)
#     curErr = prevErr + 5
#
#     while ((prevErr - curErr)**2) > epsilon:
#         prevErr = curErr
#         for i in range(0, len(centroids)):  # goes through all centroids
#             newCentroid = 0
#             sumOfProbsInRange = 0
#             for z in np.nditer(numericalVals):
#                 if bounds[i] <= z <= bounds[i + 1]:
#                     newCentroid += (z * probs[z])
#                     sumOfProbsInRange += probs[z]
#             centroids[i] = newCentroid / sumOfProbsInRange if sumOfProbsInRange != 0 else 0
#             centroids = np.sort(centroids)
#         bounds = calcBounds(k, bounds, centroids)
#         curErr = calcErr(centroids, bounds, numericalVals, probs)
#
#     newImage = np.copy(img)
#     for x, y, _ in np.ndindex(newImage.shape):
#         for i in range(0, len(bounds)):
#             color = newImage[x][y][0]
#             if color < bounds[i]:
#                 newImage[x][y] = np.repeat(centroids[i - 1], 3)
#                 break
#
#     return newImage

def calcBounds(k, bounds, centroids):
    for i in range(1, k):
        bounds[i] = ((centroids[i - 1] + centroids[i]) / 2.0)
    return bounds

def calcProbs(numOfPixels, appearances):
    probs = np.zeros(256, float)
    for i in range(0, len(appearances)):
        probs[i] = appearances[i] / float(numOfPixels)

    return probs

def calcErr(centroids, bounds, numericalVals, P):
    k = len(centroids)
    # calculate error
    errSum = 0
    for z in np.nditer(numericalVals):
        z = np.asscalar(z)
        for i in range(0, k):
            zi = bounds[i]
            zi1 = bounds[i + 1]
            if zi <= z <= zi1:
                temp = P[z] * ((z - centroids[i])**2)
                errSum += temp
    return errSum

# Task 3: Image histograms
# a
def getImageHistogram(img):
    numericalVal = img[:,:,0]
    histogram = np.zeros(256, int)
    for x in np.nditer(numericalVal):
        histogram[x] += 1
    return histogram

# b
def getConstrastStrechedImage(grayImg):
    histogram = getImageHistogram(grayImg)
    minVal, maxVal = getMinMaxValues(histogram)

    img = np.copy(grayImg)
    for x, y, _ in np.ndindex(grayImg.shape):
        val = grayImg[x][y][0]
        newVal = linearEnhancementOf(val, minVal, maxVal)
        img[x][y] = getGrayPixel(newVal)

    return img

def getMinMaxValues(histogram):
    for i in range(0, 256):
        if histogram[i] != 0:
            break
    minVal = i

    for i in range(0, 256):
        if histogram[255 - i] != 0:
            break
    maxVal = 255 - i

    return minVal, maxVal

def linearEnhancementOf(val, minVal, maxVal):
    return (val - minVal) * 255.0 / float(maxVal - minVal)

def getGrayPixel(val):
    val = val if val < 256 else 255
    return np.repeat(val, 3)

# c
def getHistEqImage(img):
    # Compute a scaling factor, α= 255 / num of pixels
    numOfPixels = np.size(img) / 3.0
    a = 255.0 / numOfPixels

    # Calculate histogram of the image
    histogram = getImageHistogram(img)

    # Create a look up table
    # LUT is Cb^-1
    LUT = np.zeros(256)
    LUT[0] = a * histogram[0]
    # for all remaining grey levels: LUT[i] = LUT[i-1] + α * histogram[i]
    for i in range(1, 256):
        val = LUT[i - 1] + (a * histogram[i])
        LUT[i] = int(val)

    # for all pixel coordinates: g(x, y) = LUT[f(x, y)] (g - new img, f - old img)
    newImg = np.copy(img)
    for x, y, _ in np.ndindex(img.shape):
        grayVal = img[x][y][0]
        val = LUT[grayVal]
        newImg[x][y] = getGrayPixel(int(val))
    return newImg

