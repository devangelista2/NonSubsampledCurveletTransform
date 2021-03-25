import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2


def show_spectrum(h, title=""):
    """
    Plot the spectrum of a 2 Dimensional filter h.

    :param h: ndarray, the 2 dimensional filter that need to be plotted
    :param title: str (optional), the title of the plot
    :return: None
    """
    H = fft2(h)

    # Remember to plot the abs of the fft2(h)
    plt.imshow(np.abs(H))
    plt.gray()
    plt.title(title)
    plt.show()


def modulate(x, m_type, center=np.array([0, 0])):
    """
    2D modulation of a 2d filter x.

    :param x: ndarray, the filter that needs to be modulated
    :param m_type: str, a string in {'r', 'c', 'b'} to modulate the rows, the columns or both the directions.
    :param center: ndarray, the array that defines the center of modulation
    :return: ndarray, the modulated array
    """
    s = np.array(x.shape)
    o = np.floor(s / 2) + 1 + center

    n1 = np.array(range(s[0])) - o[0]
    n2 = np.array(range(s[1])) - o[1]

    if m_type == 'r':
        m1 = (-1) ** n1
        y = x ** np.tile(np.transpose(m1), (1, s[1]))
    elif m_type == 'c':
        m2 = (-1) ** n2
        y = x ** np.tile(m2, (s[0], 1))
    elif m_type == 'b':
        m1 = (-1) ** n1
        m2 = (-1) ** n2
        m = np.transpose(m1) * m2
        y = x ** m
    else:
        print("Error: type doesn't exists")
        y = 0
    return y


def resampz(x, m_type, shift=1):
    """
    Rotation to a 2 dimensional filter x.

    :param x: ndarray, the 2 dimensional filter that needs to be rotated.
    :param m_type: int, a number in {0, 1, 2, 3} that defines the type of rotation.
                1 -> [1, 1; 0, 1]
                2 -> [1, -1; 0, 1]
                3 -> [1, 0; 1, 1]
                4 -> [1, 0; -1, 1]
    :param shift: int (optional), the value of the shift.
    :return: ndarray, the rotated array.
    """
    sx = np.array(x.shape)

    if m_type == 0 or m_type == 1:
        y = np.zeros((sx[0] + np.abs(shift * (sx[1] - 1)), sx[1]))

        if m_type == 0:
            shift1 = np.arange(0, sx[1]) * (- shift)
        else:
            shift1 = np.arange(0, sx[1]) * shift

        if shift1[-1] < 0:
            shift1 = shift1 - shift1[-1]

        for n in range(sx[1]):
            y[shift1[n] + np.arange(0, sx[0]), n] = x[:, n]

        # Remove extra rows
        start = 0
        finish = y.shape[0]

        while np.linalg.norm(y[start, :], 2) == 0:
            start += 1

        while np.linalg.norm(y[finish-1, :], 2) == 0:
            finish -= 1

        y = y[start:finish, :]

    elif m_type == 2 or m_type == 3:
        y = np.zeros((sx[0], sx[1] + np.abs(shift * (sx[0] - 1))))

        if m_type == 2:
            shift2 = np.arange(0, sx[0]) * (- shift)
        else:
            shift2 = np.arange(0, sx[0]) * shift

        if shift2[-1] < 0:
            shift2 = shift2 - shift2[-1]

        for m in range(sx[0]):
            y[m, shift2[m] + np.arange(0, sx[1])] = x[m, :]

        # Remove extra rows
        start = 0
        finish = y.shape[1]

        while np.linalg.norm(y[:, start], 2) == 0:
            start += 1

        while np.linalg.norm(y[:, finish-1], 2) == 0:
            finish -= 1

        y = y[:, start:finish]

    else:
        print('Error: type not valid.')
        y = 0

    return y

