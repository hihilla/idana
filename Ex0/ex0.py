import numpy as np

# Task 1: Some numerical programming in python

# 1 a
def retrunRandomMatrixWithMinMax(N):
    matrix = np.random.uniform(0, 1, (N, N))

    minimum = matrix.min()
    maximum = matrix.max()
    argmin = matrix.argmin()
    minInd = (int(argmin / N), argmin % N)
    argmax = matrix.argmax()
    maxInd = (int(argmax / N), argmax % N)

    return matrix, minimum, minInd[0], minInd[1], maximum, maxInd[0], maxInd[1]

# 1 b
def cartesian2polar2D(cartezPoints):
    polarPoints = []
    for (x, y) in cartezPoints:
        r = np.sqrt(np.power(x, 2) + np.power(y, 2))
        phi = np.arctan2(y, x)
        polarPoints.append([r, phi])

    return np.matrix(polarPoints)

# Task 2: RGB to Grayscale conversion
def convertRGB2Gray(img, converMethod ="Luminosity"):
    grayImg = [[[]]]#np.empty(len(img), len(img[0]), 3)
    for i in range(0, len(img)):
        for j in range(0, len(img[i])):
            (R, G, B) = [int(value) for value in img[i][j]]
            pixel = np.array([R, G, B])
            val = 0
            if converMethod == "Lightness":
                maximum = pixel.max()
                minimum = pixel.min()
                val = float((minimum + maximum) / 2.0)
            elif converMethod == "Average":
                val = (R + G + B) / 3.0
            else:
                val = 0.21 * R + 0.72 * G + 0.07 * B
            grayImg.append(createPixelWithValue(val))

    grayImg = np.array(grayImg, dtype=float)
    return grayImg

# Task 3: RGB to YIQ color space conversion

# 3 a
def rgb2yiq(img):
    yiqImg = np.empty((len(img), len(img[0]), 3), float)
    transformMatrix = np.array([[0.299, 0.587, 0.114],
                                [0.596, -0.274, -0.322],
                                [0.211, -0.523, 0.312]])
    for i in range(0, len(img)):
        for j in range(0, len(img[i])):
            pixel = img[i][j]
            val = np.dot(transformMatrix, pixel)
            yiqImg[i][j] = val

    return yiqImg

def yiq2rgb(img):
    rgbImg = np.empty((len(img), len(img[0]), 3), float)
    transformMatrix = np.array([[1, 0.956, 0.621],
                                [1, -0.272, -0.647],
                                [1, -1.106, 1.703]])
    for i in range(0, len(img)):
        for j in range(0, len(img[i])):
            pixel = img[i][j]
            val = np.dot(transformMatrix, pixel)
            rgbImg[i][j] = val

    return rgbImg

def createPixelWithValue(val):
    return [val, val, val]
