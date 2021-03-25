import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from scipy.signal import convolve, convolve2d, fftconvolve
from scipy.fft import fft
from skimage import data
import filters
import utils
import transform

# Define image data x
x = data.camera()
M, N = x.shape

# Define pyramidal filters
h0, h1, g0, g1 = filters.get_pyramidal_filter("9-7")

# Define Directional Filter Banks (fan filters)
u0, u1, v0, v1 = filters.get_directional_filter('haar')
u0 = utils.modulate(u0, 'c')
u1 = utils.modulate(u1, 'c')
v0 = utils.modulate(v0, 'c')
v1 = utils.modulate(v1, 'c')

# Get Parallelogram and Fan filters from fan filters
y0, y1 = filters.get_parallelogram_filters(u0, u1)

# Filter the image x in the two components
xl = convolve2d(x, h0)
xh = convolve2d(x, h1)

# Show the two components
plt.subplot(1, 2, 1)
plt.imshow(xl)
plt.title("Low pass")
plt.gray()

plt.subplot(1, 2, 2)
plt.imshow(xh)
plt.title("High pass")
plt.gray()

plt.show()

