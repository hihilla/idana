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

# Task 2: Quantization
def optimalQuantizationImage(img, k):

    pixelsNum = np.size(img) / 3.0  # maybe because its gray there's no need to divide

    epsilon = 0.05

    # choosing centroids
    centroids = np.random.randint(0, 256, k)
    centroids = np.sort(centroids)

    # bounds the centroids: the middle between two adjacent centroids
    bounds = np.zeros(k + 1, int)  # there's a chance it should be float
    bounds[0] = 0
    bounds[k] = 255
    bounds = calcBounds(k, bounds, centroids)
    probs = calcProbs(pixelsNum, getImageHistogram(img))
    clusters = calcClusters(img, bounds, k)
    you = 0

    while calcError(k, centroids, probs, clusters) < epsilon and you < 15:
        print(calcError(k, centroids, probs, clusters))
        print("hey")
        print(you)
        you += 1
        for i in range(0, len(centroids)):  # goes through all centroids
            newCentroid = 0
            sumOfProbsInRange = 0
            for grayVal in range(bounds[i], bounds[i + 1]):  # maybe +1 ? only cares about amounts, not
                                                            # actual pixels!
                newCentroid += grayVal * probs[grayVal]
                sumOfProbsInRange += probs[grayVal]
            centroids[i] = int(newCentroid / (sumOfProbsInRange + 1))
            centroids = np.sort(centroids)
        bounds = calcBounds(k, bounds, centroids)
        clusters = calcClusters(img, bounds, k)

    newImage = np.copy(img)

    for pixel in np.nditer(newImage):
        for i in range(0, len(clusters)):
            if pixel in clusters[i]:
                grayValue = centroids[i]
                pixel = np.repeat(grayValue, 3)
                break

    return newImage

def calcBounds(k, bounds, centroids):
    for i in range(1, k):
        bounds[i] = int((centroids[i - 1] + centroids[i]) / 2.0)
    return bounds

def calcClusters(image, bounds, k):
    clusters = np.empty((k, 0)).tolist()
    for pixel in np.nditer(image):
        for i in range(0, len(bounds) - 1):
            if bounds[i] < pixel < bounds[i + 1]:
                clusters[i].append(pixel)
                break
    
    return clusters

def calcProbs(pixelsNum,appearances):
    probs = np.zeros(256, float)
    for i in range(0, len(appearances)):
        probs[i] = appearances[i] / pixelsNum

    return probs

def calcError(k, centroids, probs, clusters):
    """k is number of centroids,
    centroids is array that holds centroid gray value,
    propbs is array that holds the probability of getting each gray value (0-255),
    clusters is array that holds k arrays (for k centroids):
        each sub array i will hold the pixels gray value that belong to cluster i.
    """
    sum = 0.0
    for i in range(0, k):
        xs = clusters[i]
        for x in xs:
            temp = float(probs[x] * ((x - centroids[i])**2))
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
        val = LUT[i - 1] + int(a * histogram[i])
        LUT[i] = val

    # for all pixel coordinates: g(x, y) = LUT[f(x, y)] (g - new img, f - old img)
    newImg = np.copy(img)
    for x, y, _ in np.ndindex(img.shape):
        grayVal = img[x][y][0]
        val = LUT[grayVal]
        newImg[x][y] = val
    return newImg


def foo(imageName):
    img = cv2.imread(imageName)

    qImg = optimalQuantizationImage(img, 64)
    f, ((ax1, ax2)) = plt.subplots(1, 2, sharex='col', sharey='row')

    ax1.imshow(img, cmap='gray'), ax1.set_title('Original')
    ax2.imshow(qImg, cmap='gray'), ax2.set_title('Quantized: 64')
    plt.show()


imageName = './Images/cameraman.jpg'
foo(imageName)


