import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from scipy.signal import convolve, convolve2d, fftconvolve
from scipy.fft import fft
from skimage import data
import filters
import utils


def NSLP_DEC(x, h0, h1, level):
    """
    Computes the NonSubsampled Laplacian Pyramid of an image x, with low-pass filter h0 and high-pass filter h1, at a
    fixed level.

    :param x: ndarray, the input image
    :param h0: ndarray, the low-pass filter
    :param h1: ndarray, the high-pass filter
    :param level: int, the level
    :return: tuple, a tuple (x_lo, x_hi) that contains the low/high frequency decomposition of x
    """
    if level > 0:
        I2 = np.eye(2)
        shift = -2**(level - 1) * np.array([[1, 1]]) + 2
        L = 2**level

        x_lo = utils.atrous_conv2(utils.symext(x, utils.upsample2df(h0, level), shift), h0, I2 * L)
        x_hi = utils.atrous_conv2(utils.symext(x, utils.upsample2df(h1, level), shift), h1, I2 * L)
    else:
        shift = (1, 1)
        x_lo = convolve2d(utils.symext(x, h0, shift), h0, 'valid')
        x_hi = convolve2d(utils.symext(x, h1, shift), h1, 'valid')

    return x_lo, x_hi


def NSDFB_DEC(x, filters, levels=0):
    """
    NonSubsampled Directional Filter Banks decomposition of an image x, with directional filters dfilters.

    :param x: ndarray, the input image
    :param filters: tuple, a tuple that contains the filters
    :param levels: int, the number of levels of decomposition
    :return: tuple, a tuple of subbands
    """
    if levels == 0:
        return x

    elif levels > 0:
        u0, u1, y0, y1 = filters

        # Quincunx sampling matrices
        q1 = np.array([[1, -1], [1, 1]])


def NSSFB_DEC(x, h0, h1, M):
    """
    Computes the two-channel nonsubsampled filter bank decomposition of x using the 2 dimensional filter h0 for the
    first branch, the 2 dimensional filter h1 for the second branch and the matrix M as the upsampling matrix.

    :param x: ndarray, the input image
    :param h0: ndarray, the first branch 2 dimensional filter
    :param h1: ndarray, the second branch 2 dimensional filer
    :param M: ndarray, the 2x2 upsampling matrix
    :return: tuple, a tuple (x_lo, x_hi) with the decomposition in the two branch of x
    """



