import numpy as np
from numpy.fft import fft2, ifft2, fftshift

def gaussfft(pic, t):
    Fhat = fft2(pic)
    width, height = np.shape(pic)
    xRange = range(int(-width / 2), int(-width / 2) + width)
    yRange = range(int(-height / 2), int(-height / 2) + height)
    x, y = np.meshgrid(xRange, yRange)
    gaussianFilter = (1 / (2 * np.pi * t)) * np.exp(-(x * x + y * y) / (2 * t))
    gaussianFilter = gaussianFilter / sum(sum(gaussianFilter))
    filterhat = abs(fft2(gaussianFilter))

    res = Fhat * filterhat
    return abs(ifft2(res))
