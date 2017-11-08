import numpy as np

def convertRGB2Gray(img, conversionMethod):
    col = img.shape(1)
    row = img.shape(0)
    if conversionMethod == "Lightness":
        for i in range(0, row):
            for j in range(0, col):
                min = np.amin(img[i][j])
                max = np.amax(img[i][j])
                img[i][j] = helper((min + max) / 2)

    elif conversionMethod == "Average":
            for i in range(0, row):
                for j in range(0, col):
                    avg = map(sum, img[i][j])
                    avg /= 3
                    img[i][j] = helper(avg)
    else:
        for i in range(0, row):
            for j in range(0, col):
                lum = 0.21 * img[i][j] + 0.72 * img[i][j] + 0.07 * img[i][j]
                img[i][j] = helper(lum)

    img = img[:, :, [2, 1, 0]]

def rgb2yiq(img):
    yiqConverter = np.array([[0.299, 0.587, 0.114],
                           [0.596, -0.274, -0.322]
                           [0.211, -0.523, 0.312]])
    row = img.shape(0)
    col = img.shape(1)
    for i in range(0, row):
        for j in range(0, col):
            img[i][j] = yiqConverter.dot(img[i][j])

def yiq2rgb(img):
    rgbConverter = np.array([[1, 0.965, 0.621],
                             [1, -0.272, -0.647],
                             [1, -1.106, 1.703]])
    row = img.shape(0)
    col = img.shape(1)
    for i in range(0, row):
        for j in range(0, col):
            img[i][j] = rgbConverter.dot(img[i][j])


def helper(num):
    return (num, num, num)


