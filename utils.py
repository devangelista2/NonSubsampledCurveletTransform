import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2
from scipy.signal import convolve2d


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


def mcclellan_transform(b, t): # NEEDS TO BE FIXED
    """
    Computes McClellan transform of the 1 dimensional FIR filter B with the Transformation T.

    :param b: ndarray, 1 dimensional FIR filterd B that need to be transformed
    :param t: ndarray, the transformation
    :return: ndarray, the transformed filter
    """
    # Convert the 1-D filter b to SUM_n a(n) cos(wn) form
    n = (b.shape[1] - 1) // 2
    b = np.rot90(np.fft.fftshift(np.rot90(b, 2)), 2)
    b1 = 2 * b[:, 1:n+1]
    a = np.concatenate((np.array([[b[0, 0]]]), b1), axis=1)

    inset = np.floor((np.array(t.shape) - 1) / 2)

    # Chebyshev Polynomial to compute h
    P0 = 1
    P1 = t
    h = np.array([a[:, 1]]) * P1
    rows = int(inset[0])
    cols = int(inset[1])
    h[rows, cols] = h[rows, cols] + a[:, 0] * P0

    rows = np.array([rows])
    cols = np.array([cols])
    for i in range(2, n+1):
        P2 = convolve2d(t, P1)
        rows = rows + inset[0]
        cols = cols + inset[1]

        rows = rows.astype(int)
        cols = cols.astype(int)

        P2 = P2[rows, :]
        P2 = P2[:, cols]
        P2 = P2 - P0

        rows = inset[0] + np.arange(0, P1.shape[0])
        cols = inset[1] + np.arange(0, P1.shape[1])

        rows = rows.astype(int)
        cols = cols.astype(int)

        hh = h
        h = a[:, i] * P2

        h = h[rows, :]
        h = h[:, cols]
        h = h + hh

        P0 = P1
        P1 = P2

    return h


def symext(x, h, shift=(1, 1)):
    """
    Performes symmetric extension of an image x and a 2 dimensional filter h. The filer is assumed to have odd dimension

    :param x: ndarray, the input image
    :param h: ndarray, the 2 dimensional filter
    :param shift: tuple or list, the shift of the filter
    :return: ndarray, the filtered image
    """
    m, n = x.shape
    p, q = h.shape

    p2 = int(np.floor(p/2))
    q2 = int(np.floor(q/2))

    s1, s2 = shift

    ss = p2 - s1 + 1
    rr = q2 - s2 + 1

    y = np.concatenate((np.fliplr(x[:, :ss]), x, x[:, n:n-p-s1+1:-1]), axis=1)
    y = np.concatenate((np.flipud(y[:rr, :]), y, y[m:m-q-s2+1:-1, :]), axis=0)
    y = y[:m+p-1, :n+q-1]

    return y


def atrous_conv2(x, h, S):
    """
    Computes atrous convolution for a 2x2 diagonal matrix S, where the 2 dimensional filter h is applied to x with
    trous.

    (very slow)

    :param x: ndarray, the input image
    :param h: ndarray, the 2 dimensional FIR filter
    :param S: ndarray, the 2x2 upsampling matrix
    :return: ndarray, the convolved image
    """
    M, N = x.shape
    P, Q = h.shape

    S0 = S[0, 0]
    S1 = S[1, 1]

    # Convert to integers
    M = int(M)
    N = int(N)
    P = int(P)
    Q = int(Q)
    S0 = int(S0)
    S1 = int(S1)

    SP = P - 1
    SQ = Q - 1

    SS0 = S0 - 1
    SS1 = S1 - 1

    # Compute output shapes
    MM = M - S0*P + 1
    NN = N - S1*Q + 1

    y = np.zeros((MM, NN))

    # Convoluion loop
    for i in range(MM):
        for j in range(NN):
            ssum = 0
            kk1 = i + SS0
            for k in range(P):
                kk2 = j + SS1
                for t in range(Q):
                    f1 = SP - k
                    f2 = SQ - t
                    ssum = ssum + h[f1, f2] * x[kk1, kk2]
                    kk2 = kk2 + S0
            y[i, j] = ssum

    return y


def upsample2df(h, p):
    """
    Upsample a filter by 2^p.

    :param h: ndarray, the 2 dimensional filter
    :param p: int, the power of 2
    :return: ndarray, the upsampled filter
    """
    m, n = tuple(map(int, h.shape))
    g = np.zeros((2**p * m, 2**p * n))
    g[::2**p, ::2**p] = h

    return g


def efilter2(x, h, extmod='per', shift=(0, 0)):
    """
    Computes 2D filtering with edge handling (via extension)

    :param x: ndarray, the input image
    :param h: ndarray, the 2 dimensional filter
    :param extmod: str (optional), a string that describes the type of extension
    :param shift: list or tuple (optional), a tuple (sx, sy) that represents the shift
    :return: ndarray, the filtered image
    """
    sf = (np.array(h.shape) - 1) // 2

    ru = int(np.floor(sf[0]) + shift[0])
    rd = int(np.ceil(sf[0]) + shift[0])
    cl = int(np.floor(sf[1]) + shift[1])
    cr = int(np.ceil(sf[1]) + shift[1])

    x_ext = extend2(x, ru, rd, cl, cr, extmod)
    y = convolve2d(x_ext, h, 'valid')

    return y


def extend2(x, ru, rd, cl, cr, extmod):
    """
    Computes the 2 dimensional extension of x.
    Supported extensions:
        - per -> periodic extension
        - qper_row
        - qper_col

    :param x: ndarray, the input image
    :param ru: int, amount of extension (up) for the rows
    :param rd: int, amount of extension (down) for the rows
    :param cl: int, amount of extension (left) for the cols
    :param cr: int, amount of extension (right) for the cols
    :param extmod: str, extension mode
    :return: ndarray, the extended array
    """
    rx, cx = int(x.shape[0]), int(x.shape[1])

    if extmod == "per":
        I = get_permutation_indices(rx, ru, rd)
        y = x[I, :]

        I = get_permutation_indices(cx, cl, cr)
        y = x[:, I]

    elif extmod == "qper_row":
        rx2 = int(round(rx / 2))

        y1 = x[rx2:rx, cx-cl:cx]
        y2 = x[:rx2, cx-cl:cx]
        y3 = x[rx2:rx, :cr]
        y4 = x[:rx2, :cr]

        y5 = np.concatenate((y1, y2))
        y6 = np.concatenate((y3, y4))

        y = np.concatenate((y5, x, y6), axis=1)

        I = get_permutation_indices(rx, ru, rd)
        y = y[I, :]

    elif extmod == "qper_col":
        cx2 = int(round(cx / 2))

        y1 = x[rx-ru:rx, cx2:cx]
        y2 = x[rx-ru:rx, :cx2]
        y3 = x[:rd, cx2:cx]
        y4 = x[:rd, :cx2]

        y5 = np.concatenate((y1, y2), axis=1)
        y6 = np.concatenate((y3, y4), axis=1)

        y = np.concatenate((y5, x, y6), axis=0)

        I = get_permutation_indices(cx, cl, cr)
        y = y[:, I]

    else:
        y = 0
        print("Error! Extmod not available")

    return y


def get_permutation_indices(lx, lb, le):
    I1 = np.arange(lx - lb, lx)
    I2 = np.arange(0, lx)
    I3 = np.arange(0, le)

    I = np.concatenate((I1, I2, I3))

    if (lx < lb) or (lx < le):
        I = np.mod(I, lx)
        I[I == 0] = lx

    return I


def zconv2(x, h, S):
    """
    Computes atrous convolution for a 2x2 non-diagonal matrix S, where the 2 dimensional filter h is applied to x with
    trous.

    :param x: ndarray, the input image
    :param h: ndarray, the 2 dimensional FIR filter
    :param S: ndarray, the 2x2 upsampling matrix
    :return: ndarray, the convolved image
    """
    M, N = int(x.shape[0]), int(x.shape[1])
    P, Q = int(h.shape[0]), int(h.shape[1])

    M0 = int(S[0, 0])
    M1 = int(S[0, 1])
    M2 = int(S[1, 0])
    M3 = int(S[1, 1])

    NewP = ((M0-1)*(P-1))+M2*(Q-1) + P-1
    NewQ = ((M3-1)*(Q-1))+M1*(P-1) + Q-1

    y = np.zeros((M, N))

    ssum = 0
    Start1 = NewP // 2
    Start2 = NewQ // 2
    mn1 = Start1 % M
    mn2 = Start2 % N
    mn2_save = mn2

    for n1 in range(M):
        for n2 in range(N):
            outindexx = mn1
            outindexy = mn2

            for l1 in range(P):
                indexx = outindexx
                indexy = outindexy
                for l2 in range(Q):
                    ssum += x[indexx, indexy] * h[l1, l2]
                    indexx = indexx - M2

                    if indexx < 0:
                        indexx = indexx + M
                    if indexx > M-1:
                        indexx = indexx - M
                    indexy = indexy - M3
                    if indexy < 0:
                        indexy = indexy + N

                outindexx = outindexx - M0
                if outindexx < 0:
                    outindexx = outindexx + M

                outindexy = outindexy - M1
                if outindexy < 0:
                    outindexy = outindexy + N
                if outindexy > N-1:
                    outindexy = outindexy - N

            y[n1, n2] = ssum
            ssum = 0
            mn2 = mn2 + 1

            if mn2 > N-1:
                mn2 = mn2 - N

        mn2 = mn2_save
        mn1 = mn1 + 1
        if mn1 > M-1:
            mn1 = mn1 - M

    return y
