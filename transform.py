import numpy as np
from scipy.signal import convolve2d, fftconvolve
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
        # No filters application
        return x

    elif levels > 0:
        u0, u1, p0, p1 = filters

        # Quincunx sampling matrices
        q1 = np.array([[1, -1], [1, 1]])

        if levels == 1:
            # No Upsampling filters at the first level
            y1, y2 = NSSFB_DEC(x, u0, u1)
            y = (y1, y2)

        else:
            # No Upsampling filters at the first level
            x1, x2 = NSSFB_DEC(x, u0, u1)

            # Convolution with Upsampled Filters
            y1, y2 = NSSFB_DEC(x1, u0, u1, q1)
            y3, y4 = NSSFB_DEC(x2, u0, u1, q1)

            y = (y1, y2, y3, y4)

            # Higher level of decomposition
            for l in range(3, levels):
                y_old = y
                y = tuple()

                # First channel
                for k in range(2**(l-2)):
                    # Compute the upsampling matrix by the formula (3.18) of Mihn N. Do' thesis. The upsampling matrix
                    # For the channel k in an l-levels DFB is
                    # M_k^{(l-1)} (refer to pp. 53, (3.18), Mihn N. Do's thesis).

                    # Compute s_k^{(l-1)}
                    s_k = 2 * np.floor(k / 2) - 2 ** (l-3) + 1
                    # Compute the sampling matrix
                    m_k = 2 * np.dot(np.array([[2**(l-3), 0], [0, 1]]), np.array([[1, 0], [-s_k, 1]]))
                    i = (k-1) % 2
                    # Decompose
                    y1, y2 = NSSFB_DEC(y_old[k], y3[i], y3[i], m_k)
                    y = y + (y1, y2)

                # Second channel
                for k in range(2**(l-2)):
                    # Compute the upsampling matrix by the formula (3.18) of Mihn N. Do' thesis. The upsampling matrix
                    # For the channel k in an l-levels DFB is
                    # M_k^{(l-1)} (refer to pp. 53, (3.18), Mihn N. Do's thesis).

                    # Compute s_k^{(l-1)}
                    s_k = 2 * np.floor(k / 2) - 2 ** (l - 3) + 1
                    # Compute the sampling matrix
                    m_k = 2 * np.dot(np.array([[1, 0], [0, 2**(l-3)]]), np.array([[1, -s_k], [0, 1]]))
                    i = (k - 1) % 2
                    # Decompose
                    y1, y2 = NSSFB_DEC(y_old[k + 2**(l-2)], y4[i], y4[i], m_k)
                    y = y + (y1, y2)

    return y


def NSSFB_DEC(x, h0, h1, M=np.eye(2)):
    """
    Computes the two-channel nonsubsampled filter bank decomposition of x using the 2 dimensional filter h0 for the
    first branch, the 2 dimensional filter h1 for the second branch and the matrix M as the upsampling matrix.

    :param x: ndarray, the input image
    :param h0: ndarray, the first branch 2 dimensional filter
    :param h1: ndarray, the second branch 2 dimensional filer
    :param M: ndarray, the 2x2 upsampling matrix
    :return: tuple, a tuple (x_lo, x_hi) with the decomposition in the two branch of x
    """
    if type(M) == int or (M == np.eye(2)).all():
        # No Sampling
        y1 = utils.efilter2(x, h0)
        y2 = utils.efilter2(x, h1)

    elif M.shape == (2, 2):
        # Non-Separable 2x2 Sampling Matrix
        y1 = utils.zconv2(x, h0, M)
        y2 = utils.zconv2(x, h1, M)

    elif M.shape == (1, 1):
        # Separable 2x2 Sampling Matrix
        M = M * np.eye(2)
        y1 = utils.zconv2(x, h0, M)
        y2 = utils.zconv2(x, h1, M)

    else:
        print("Error in the matrix M: unaccepted shape.")
        y1 = 0
        y2 = 0

    return y1, y2


def NSLP_REC(y, g0, g1):
    """
    Computes the inverse NonSubsampled Laplacian Pyramid of y, given the reconstruction filters g0, g1.

    :param y: tuple, a tuple of length L+1 that contains the high pass bands at each decomposition level
    :param g0: ndarray, the lowpass reconstruction filter
    :param g1: ndarray, the highpass reconstruction filter
    :return: ndarray, the reconstructed image
    """
    x_hi, x_lo = y
    shift = (1, 1)
    x = convolve2d(utils.symext(x_lo, g0, shift), g0, 'valid') + convolve2d(utils.symext(x_hi, g1, shift), g1, 'valid')
    return x

