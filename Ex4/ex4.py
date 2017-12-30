import numpy as np


# Task 2
# a
def Fourier1D(x_n):
    n = len(x_n)
    x_n_fourier = np.ndarray(n, complex)
    exponents = np.ndarray(n)
    exponents = np.repeat(-2 * np.pi / n, n)
    x_nIndices = np.arange(n)
    exponents = np.multiply(exponents, x_nIndices)

    # for each element in the transformed vector, the following calcs the
    # power of the exponent, calcs the imaginary and real values and sums
    # them to their right position of the transformed vector
    for k in np.arange(0, n):
        # calculate the real and imaginary values
        # the 2 val arrays will hold each iteration of the sum (sigma)
        realVals = np.ndarray(n)
        imagVals = np.ndarray(n, complex)
        for i in np.arange(0, n):
            realVals[i] = np.cos(np.pi * exponents[i])
            imagVals[i] = 1j * np.sin(np.pi * exponents[i])
        realVals = np.multiply(realVals, x_n)
        imagVals = np.multiply(imagVals, x_n)
        x_n_fourier[k] = np.sum(realVals) + np.sum(imagVals)

    return x_n_fourier

# b
def invFourier1D(F_n):
    n = len(F_n)
    x_n_invFourier = np.ndarray(n)

    # for every element of the original (unknown) vector calculates:
    # the exponents and their cos ans sin (real and imaginary)
    # sums both array of results and devides to real and imaginary parts
    for x in np.arange(0, n):
        exponents = np.ndarray(n)
        k_index = 0
        # exponents powers calculations
        for k in np.nditer(F_n):
            exponents[k_index] = 2 * np.pi * k_index * x / n
            k_index += 1
        cosVals = np.ndarray(n, complex)
        sinVals = np.ndarray(n, complex)
        for i in np.arange(0, n):
            cosVals[i] = np.cos(exponents[i])
            sinVals[i] = 1j * np.sin(exponents[i])
        realAndImagSum = np.sum(cosVals) + np.sum(sinVals)
        x_n_invFourier[x] = realAndImagSum

    return x_n_invFourier


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
