import numpy as np


# Task 2
# a
def Fourier1D(x_n):
    n = len(x_n)
    x_n_fourier = np.ndarray(n, complex)

    # for each element in the transformed vector, the following calcs the
    # power of the exponent, calcs the imaginary and real values and sums
    # them to their right position of the transformed vector
    for k in np.arange(0, n):
        exponents = np.ndarray(n)
        # calculates the exponents power only
        for x in np.arange(n):
            exponents[x] = -2 * k * x / n

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

    zeroPaddedFFT = np.lib.pad(shiftFFT, padWidth, 'constant')
    zeroPaddedFFT *= (upsamplingFactor[0] * upsamplingFactor[1])

    # Shift the High frequency components to the center and Low frequency components outside.
    invShift = np.fft.ifftshift(zeroPaddedFFT)

    # Perform Inverse Fast Fourier Transform
    # upsampledImg = np.array(np.fft.ifft2(invShift).real, dtype=int)
    upsampledImg = np.abs(np.fft.ifft2(invShift))
    return upsampledImg, shiftFFT, zeroPaddedFFT


# Task 4
def phaseCorr(img1, img2):
    """
    Implement an algorithm that finds a translation between two images sampled from an original
    image. this, implement the phase-correlation algorithm which use the frequency domain to find
    the translation in x, y between two images.
    :param img1:
    :param img2:
    :return:
    """
    return 0


# Task 5
def imFreqFilter(img, lowThresh, highThresh):
    return 0


# d
def imageDeconv(imgMotion, kernel_motion_blur, k):
    return 0


def imageDeconv(imgEcho, kernel_echo_blur, k):
    return 0
