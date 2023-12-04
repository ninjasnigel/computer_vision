import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import color

def pflat(x):
    return x[:-1,:] / x[-1,:]

# Load data from data/compEx2.jpg, we need to make the image grayscale
im = plt.imread('data/compEx2.jpg')

# Load data from data/compEx2.mat
mat = scipy.io.loadmat('data/compEx2.mat')

# Extract the points from the data
p1, p2, p3 = mat['p1'], mat['p2'], mat['p3']

# Plot the points
plt.imshow(im, cmap='gray')
plt.plot(p1[0], p1[1], 'r.')
plt.plot(p2[0], p2[1], 'g.')
plt.plot(p3[0], p3[1], 'b.')

def point_line_distance_2D(x, l):
    """
    Computes the distance between a 2D point and a line.

    :param x: A tuple or array representing the point (x1, x2).
    :param l: A tuple or array representing the line (a, b, c) as in the line equation ax + by + c = 0.
    :return: The distance from the point to the line.
    """
    x1, x2 = x
    a, b, c = l

    # Calculate the distance using the formula
    distance = abs(a * x1 + b * x2 + c) / np.sqrt(a**2 + b**2)
    return distance

# Calculate the x-limits of the image
x_limits = [0, im.shape[1]]

lines = []

for p, color in zip([p1, p2, p3], ['r', 'g', 'b']):
    # Calculate slope (m) and y-intercept (c)
    m = (p[1, 1] - p[1, 0]) / (p[0, 1] - p[0, 0])
    c = p[1, 0] - m * p[0, 0]

    lines.append([m, c])

    # Calculate y-values at the x-limits
    y_extended = m * np.array(x_limits) + c

    # Plot the extended line
    plt.plot(x_limits, y_extended, color + '-')

# Calculate the intersection points
m1, c1 = lines[0]
m2, c2 = lines[1]
m3, c3 = lines[2]
x_intersect = (c3 - c2) / (m2 - m3)
y_intersect = m2 * x_intersect + c2

point = (x_intersect, y_intersect)
line = line = (m1, -1, c1)

# Plot the intersection point
plt.plot(x_intersect, y_intersect, 'k.', markersize=10)

#compute the distance between the first line and the the intersection point.

distance = point_line_distance_2D(point, line)

print("The distance between the first line and the intersection point is: ", distance)

plt.show()
