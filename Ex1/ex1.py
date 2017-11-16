import cv2
import numpy as np

# Task 1: Spatial sampling
# a
def getSampledImageAtResolution(dim, pixelSize, x):
    return 0


# b

# Task 2: Quantization
def optimalQuantizationImage(img, k):
    Q = np.random.randint(0, 256, k)
    Q = np.sort(Q)
    epsilon = 0.003
    bounds = np.zeros(k + 1, int) #there's a chance it should be float
    for i in range(0, k + 1):
        bounds[i] = (Q[i + 1] - Q[i]) / 2

    while (calcError(k, Q,  ))

    return 0

#gets the image, returns array with # of appearances of each pixel [0-255]
def countApperances(img):
    numericalVal = cv2.imread(img)
    appearances = np.zeros(256, int)

    for x in np.nditer(numericalVal):
        appearances[x] += 1

    return appearances

def calcError(k, centroids, appearances):
    for i in range(0,len(centroids)):

    return 0



# Task 3: Image histograms
# a
def getImageHistogram(img):
    return 0

# b
def getConstrastStrechedImage(img):
    return 0

# c
def getHistEqImage(img):
    return 0