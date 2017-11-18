import numpy as np
import cv2

# Task 1: Spatial sampling
def getSampledImageAtResolution(dim, pixelSize, k=2):
    x0 = dim[0]
    x1 = dim[1]
    y0 = dim[2]
    y1 = dim[3]

    numXPixels = int(float(x1 - x0) / pixelSize)
    numYPixels = int(float(y1 - y0) / pixelSize)

    X = []
    Y = []

    for i in range(0, numXPixels):
        val = x0 + (i * pixelSize)
        X.append(val)

    X = np.array(X, float)
    X = np.multiply(3, X)

    for i in range(0, numYPixels):
        val = y0 + (i * pixelSize)
        Y.append(val)

    Y = np.array(Y, float)
    Y = np.multiply(2, Y)

    Xs, Ys = np.meshgrid(X, Y)
    return np.cos(k * np.pi * (Xs + Ys))

# def analyticFunction(x, y, k):
#     x = np.multiply(3, x)
#     y = np.multiply(2, y)
#
#     tempMatrix = np.add(x, y)
#
#     tempParam = k * np.pi
#     tempMatrix = np.multiply(tempParam, tempMatrix)
#     return np.cos(tempMatrix)


# Task 2: Quantization
def optimalQuantizationImage(img, k):
    image = cv2.imread(img)
    pixelsNum = image.size / 3 #maybe because its gray there's no need to divide

    epsilon = 0.003

    # choosing centroids
    Q = np.random.randint(0, 256, k)
    Q = np.sort(Q)
    # bounds the centroids: the middle between two adjacent centroids
    bounds = np.zeros(k + 1, int) #there's a chance it should be float
    for i in range(0, k + 1):
        bounds[i] = (Q[i + 1] - Q[i]) / 2

    probs = calcProbs(pixelsNum, getImageHistogram(img))
    while (calcError(k, Q, probs) < epsilon):
        print("nothing")


    return 0

def calcProbs(pixelsNum,appearances):
    probs = np.zeros(256, float)
    for i in range(0, appearances):
        probs[i] = appearances[i] / pixelsNum

    return probs

def calcError(k, centroids, probs, clusters):
    """k is number of centroids,
    centroids is array that holds centroid gray value,
    propbs is array that holds the probability of getting each gray value (0-255),
    clusters is array that holds k arrays (for k centroids):
        each sub array i will hold the pixels ([x][y][gray value]) that belong to cluster i.
    """
    sum = 0
    for i in range(0, k):
        xs = clusters[i]
        for x in xs:
            grayColor = x[2]
            temp = probs[grayColor] * ((grayColor - centroids[i])**2)
            sum += temp
    return sum

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
    minVal, maxVal = findMinMaxValues(histogram)
    img = np.copy(grayImg)

    for x, y, _ in np.ndindex(grayImg.shape):
        val = grayImg[x][y][0]
        newVal = linearEnhancementOf(val, minVal, maxVal)
        img[x][y] = getGrayPixel(newVal)
    return img

def findMinMaxValues(histogram):
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
    shape = np.shape(img)
    numOfPixels = np.size(img) / 3.0
    a = 255.0 / numOfPixels

    # Calculate histogram of the image
    histogram = getImageHistogram(img)

    # Create a look up table
    LUT = np.zeros(256)
    # LUT[0] =  α * histogram[0]
    LUT[0] = a * histogram[0]

    # for all remaining grey levels: LUT[i] = LUT[i-1] + α * histogram[i]
    for i in range(1, 256):
        val = LUT[i - 1] + a * histogram[i]
        LUT[i] = val

    # for all pixel coordinates: g(x, y) = LUT[f(x, y)]
    image = np.copy(img)
    for x, y, _ in np.ndindex(img.shape):
        grayVal = img[x][y][0]
        val = LUT[f(grayVal, numOfPixels, histogram)]
        image[x][y] = val
    return image

def getCumulativeDistribution(histogram):
    cdf = np.copy(histogram)
    # cdf.append(histogram[0])
    for i in range(1, 256):
        c = cdf[i - 1] + histogram[i]
        cdf[i] = c
    return np.array(cdf)

def f(grayVal, imgSize, histogram):
    # v = img[x][y][0] # gray value
    cdf = getCumulativeDistribution(histogram)
    cdfMin, _ = findMinMaxValues(cdf)
    # imgSize = np.size(img) / 3.0
    return int(255.0 * (cdf[grayVal] - cdfMin) / (imgSize - 1))
