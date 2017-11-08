import numpy as np
import cv2
import matplotlib.pyplot as plt

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
def convertRGB2Gray(img, converMethod = "Luminosity"):
    image = np.copy(img)
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
            image[i][j] = createPixelWithValue(val)
    return image

# Task 3: RGB to YIQ color space conversion

# 3 a
def rgb2yiq(img):
    image = np.copy(img)
    transformMatrix = np.array([[0.299, 0.587, 0.114],
                                [0.596, -0.274, -0.322],
                                [0.211, -0.523, 0.312]])
    # for i in range(0, len(img)):
    #     for j in range(0, len(img[i])):
    #         pixel = img[i][j]
    #         val = np.dot(transformMatrix, pixel)
    #         image[i][j] = val
    return image.dot(transformMatrix)

def yiq2rgb(img):
    image = np.copy(img)
    transformMatrix = np.array([[0.299, 0.587, 0.114],
                                [0.596, -0.274, -0.322],
                                [0.211, -0.523, 0.312]])
    transformMatrix = transformMatrix.transpose()
    for i in range(0, len(img)):
        for j in range(0, len(img[i])):
            pixel = img[i][j]
            val = np.dot(transformMatrix, pixel)
            for k in range(0, 3):
                if val[k] < 0:
#                    print(0-val[k])
                    val[k] = 0
            image[i][j] = val
    return image#.dot(transformMatrix)

def createPixelWithValue(val):
    intval = int(val)
    return [intval, intval, intval]

def test_3(imageName):
    img = cv2.imread(imageName);

    # reorder from BGR to RGB
    img = img[:, :, [2, 1, 0]]

    # normalize to range [0 1]
    img = np.float32(img)
    img = img / 255

    img_yiq = rgb2yiq(img)
    img_rgb = yiq2rgb(img_yiq)

    f, ((ax1, ax2)) = plt.subplots(1, 2, sharex='col')
    ax1.imshow(img), ax1.set_title('Original RGB image')

imageName = './Images/peppers.png'
test_3(imageName)