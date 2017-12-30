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
        cosVals = np.cos(np.multiply(np.multiply(exponents, x), np.pi))
        sinVals = np.sin(np.multiply(np.multiply(exponents, x), np.pi))
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

    polar = np.column_stack((R, theta))

    return polar

# c
def Fourier1DPolar(x_n):

    return cartesianToPolar(Fourier1D(x_n))


# d
def invFourier1DPolar(F_n_polar):

    return cartesianToPolar(invFourier1D(F_n_polar))


# Task 3
def imageUpsampling(img, upsamplingFactor):
    """
    Implement a 2D image upsampling function. The algorithm should be the zero-padding in the
    frequency domain.
    In case of scaling factor < 1, the function should return the original image and its Fourier
    transform for both the Fourier and the zero-padded Fourier return values.
    :param img: 2D image
    :param upsamplingFactor: 2D vector with the upscaling parameters (larger than 1)
    at each dimension
    :return: The upsampled image, the FFT of the original images, and the zero-padded FFT
    """

    upsamplingFactor = np.array(upsamplingFactor)
    FFTimg = np.fft.fft2(img)
    # Shift the Low frequency components to the center and High frequency components outside.
    shiftFFT = np.fft.fftshift(FFTimg)

    # if scaling factor < 1, return original img and fourier transform
    if (upsamplingFactor < 1).any():
        return img, shiftFFT, shiftFFT

    # Pad with zeros
    shape = np.array(img.shape)
    padWidth = np.array(((shape * upsamplingFactor) - shape) / 2.0, dtype=int)
    _, padWidth = np.meshgrid(padWidth, padWidth)

    zeroPaddedFFT = np.lib.pad(shiftFFT,
                               padWidth,
                               'constant')
    zeroPaddedFFT *= (upsamplingFactor[0] * upsamplingFactor[1])

    # Shift the High frequency components to the center and Low frequency components outside.
    invShift = np.fft.ifftshift(zeroPaddedFFT)

    # Perform Inverse Fast Fourier Transform
    # upsampledImg = np.array(np.fft.ifft2(invShift).real, dtype=int)
    upsampledImg = np.abs(np.fft.ifft2(invShift))
    return upsampledImg, shiftFFT, zeroPaddedFFT


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
