import numpy as np


# Task 2
# a
def Fourier1D(x_n):
    n = len(x_n)
    x_n_fourier = np.ndarray(n)

    # for each element in the transformed vector, the following calcs the
    # power of the exponent, calcs the omaginart and real values and sums
    # them to their right position of the transformed vector
    for k in np.arange(0, n):
        exponents = np.ndarray(n)
        # calculates the exponents power only
        x_index = 0
        for x in np.nditer(x_n):
            exponents[x_index] = -2 * np.pi * k * x / n
            x_index += 1
        # calculate the real and imaginary values
        # the 2 val arrays will hold each iteration of the sum (sigma)
        realVals = np.ndarray(n)
        imagVals = np.ndarray(n)
        for i in np.arange(0, n):
            realVals[i] = np.cos(exponents[i])
            imagVals[i] = 1j * np.sin(exponents[i])
        realVals = np.multiply(realVals, x_n)
        imagVals = np.multiply(imagVals, x_n)
        x_n_fourier[k] = np.sum(realVals) + np.sum(imagVals)

    return x_n_fourier

# b
def invFourier1D(F_n):
    return 0


# c
def Fourier1DPolar(x_n):
    return 0


# d
def invFourier1DPolar(F_n_polar):
    return 0


# Task 3
def imageUpsampling(img, upsamplingFactor):
    return 0


# Task 4
def phaseCorr(img1, img2):
    return 0


# Task 5
def imFreqFilter(img, lowThresh, highThresh):
    return 0


# d
def imageDeconv(imgMotion, kernel_motion_blur, k):
    return 0


def imageDeconv(imgEcho, kernel_echo_blur, k):
    return 0
