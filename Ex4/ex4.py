import numpy as np


# Task 2
# a
def Fourier1D(x_n):
    n = len(x_n)
    x_n_fourier = np.ndarray(n, complex)

    # vector of the powers of the exponent
    exponents = np.ndarray(n)
    exponents = np.repeat(-2 / n, n)
    x_nIndices = np.arange(n)
    exponents = np.multiply(exponents, x_nIndices)

    # with each iteration fills an element of the transformed vector
    for k in np.arange(0, n):
        realVals = np.ndarray(n)
        imagVals = np.ndarray(n, complex)

        # each element is the result of cos/sin of the exponent,
        # and transformed the sin values to be imaginary
        realVals = np.cos(np.multiply(exponents, np.pi * k))
        imagVals = np.sin(np.multiply(exponents, np.pi * k))
        imagVals = np.multiply(imagVals, 1j)

        realVals = np.multiply(x_n, realVals)
        imagVals = np.multiply(x_n, imagVals)

        x_n_fourier[k] = np.sum(realVals) + np.sum(imagVals)

    return x_n_fourier

# b
def invFourier1D(F_n):
    n = len(F_n)
    x_n_invFourier = np.ndarray(n, complex)

    # vector of the powers of the exponent
    exponents = np.repeat(2 / n, n)
    k_nIndices = np.arange(n)
    exponents = np.multiply(exponents, k_nIndices)

    # with each iteration fills an element of the 'original' vector
    for x in np.arange(0, n):
        cosVals = np.ndarray(n, complex)
        sinVals = np.ndarray(n, complex)

        # each element is the result of cos/sin on the exponent,
        # transformed the sin values to be imaginary
        cosVals = np.cos(np.multiply(exponents, np.multiply(np.pi, x)))
        sinVals = np.sin(np.multiply(exponents, np.multiply(np.pi, x)))
        sinVals = np.multiply(sinVals, 1j)

        cosVals = np.multiply(F_n, cosVals)
        sinVals = np.multiply(F_n, sinVals)

        x_n_invFourier[x] = np.divide(np.sum(cosVals) + np.sum(sinVals), n)

    return x_n_invFourier


def cartesianToPolar(cartesian):
    real = np.real(cartesian)
    imag = np.imag(cartesian)

    R = np.sqrt(np.power(real, 2) + np.power(imag, 2))
    theta = np.arctan2(imag, real)

    polar = np.multiply(R, np.exp(np.multiply(1j, theta)))

    return polar

# c
def Fourier1DPolar(x_n):

    return cartesianToPolar(Fourier1D(x_n))


# d
def invFourier1DPolar(F_n_polar):

    return cartesianToPolar(invFourier1D(F_n_polar))


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
