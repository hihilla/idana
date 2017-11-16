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

    bounds = np.zeros(k + 1, int) #there's a chance it should be float
    for i in range(0, k + 1):
        bounds[i] = (Q[i + 1] - Q[i]) / 2

    

    return 0

def calcError(k, *centroids, **probs):
    for i in range(0,len(centroids)):
        for j in range(0, len(centroids) + 1):




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