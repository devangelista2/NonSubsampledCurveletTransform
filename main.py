import numpy as np
import matplotlib.pyplot as plt
import math
from skimage import data
import filters
import utils
import transform

# Define image data x
x_true = data.camera()
x = x_true
M, N = x.shape

# Define pyramidal filters
h0, h1, g0, g1 = filters.get_pyramidal_filter('9-7')

h0 = h0 / np.sqrt(2)
h1 = h1 / np.sqrt(2)
g0 = g0 / np.sqrt(2)
g1 = g1 / np.sqrt(2)

x_lo, x_hi = transform.NSLP_DEC(x, h0, h1, 0)
x_rec = transform.NSLP_REC((x_hi, x_lo), g0, g1)

plt.subplot(1, 3, 1)
plt.imshow(x_true)
plt.gray()

plt.subplot(1, 3, 2)
plt.imshow(x_rec)
plt.gray()

plt.subplot(1, 3, 3)
plt.imshow(x_lo)
plt.gray()
plt.show()

"""
# Define Directional Filter Banks (fan filters)
u0, u1, v0, v1 = filters.get_directional_filter('haar')

u0 = utils.modulate(u0, 'c')
u1 = utils.modulate(u1, 'c')
v0 = utils.modulate(v0, 'c')
v1 = utils.modulate(v1, 'c')

# Get Parallelogram and Fan filters from fan filters
y0, y1 = filters.get_parallelogram_filters(u0, u1)

# Unify
dfb_filters = (u0, u1, y0, y1)

# Define parameters for the transform
levels = [0]
n_levels = len(levels)
n_index = n_levels


# Initialize the output
y = [0] * (n_levels + 1)

for i in range(n_levels):

    # Non-Subsampled Laplacian Pyramid Decomposition
    [x_lo, x_hi] = transform.NSLP_DEC(x, h0, h1, i)

    if levels[n_index - 1] > 0:
        # Nonsubsampled DFB decomposition on the bandpass image
        xhi_dir = transform.NSDFB_DEC(x_hi, dfb_filters, levels[n_index - 1])
        y[i] = xhi_dir

    else:
        y[i] = x_hi

    n_index = n_index - 1

    # Prepare the next iteration
    x = x_lo

y[n_levels] = x

# Get the components out of y
x_hi, x_lo = y[0], y[1] # In y the band frequencies are reversed

print(x_hi.shape)
print(x_lo.shape)

# Show the two components
plt.subplot(1, 2, 1)
plt.imshow(x_lo)
plt.title("Low pass")
plt.gray()

plt.subplot(1, 2, 2)
plt.imshow(x_hi)
plt.title("High pass")
plt.gray()

plt.show()

# Reconstruction
x_rec = transform.NSLP_REC((x_hi, x_lo), g0, g1)

plt.imshow(x)
plt.gray()
plt.show()
"""