import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from scipy.signal import convolve, convolve2d, fftconvolve
from scipy.fft import fft
from skimage import data
import filters
import utils


def NSCTDec(x, levels, pfilt='9-7', dfilt='haar'):
    """
    Execute Non-Subsampled Contourlet Transform on x, with a certain numbers of levels, using the filter defined in
    pfilt for the Non-Subsampled Laplacian Pyramid, the filter defined in dfilt for the Non-Subsampled Directional
    Filter Banks.

    :param x: ndarray, the input image.
    :param levels: list, a list of integer containing the levels
    :param pfilt: str, a string with the name of the NSLP filters
    :param dfilt: str, a string with the name of the NSDFB filters
    :return: list, a list of ndarray containing the coefficient of the decomposition of x
    """
    # Define pyramidal filters
    h0, h1, g0, g1 = filters.get_pyramidal_filter(pfilt)

    # Define Directional Filter Banks (fan filters)
    u0, u1, v0, v1 = filters.get_directional_filter(dfilt)

    u0 = utils.modulate(u0, 'c')
    u1 = utils.modulate(u1, 'c')
    v0 = utils.modulate(v0, 'c')
    v1 = utils.modulate(v1, 'c')

    # Get Parallelogram and Fan filters from fan filters
    y0, y1 = filters.get_parallelogram_filters(u0, u1)

    # Unify
    dfb_filters = [u0, u1, y0, y1]