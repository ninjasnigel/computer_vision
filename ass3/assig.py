import numpy as np
from scipy.linalg import null_space
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import color

import numpy as np

# Fundamental matrix F
F = np.array([[0, 1, 1], [2, 0, 4], [0, 1, 1]])

# Scene points
X1 = np.array([0, 3, 1, 1])  # Homogeneous coordinates
X2 = np.array([-1, 2, 0, 1])

# Camera matrix P1 = [I | 0]
P1 = np.hstack((np.eye(3), np.zeros((3, 1))))

# Project points in camera P1
x1 = P1 @ X1
x2 = P1 @ X2

# Normalize to homogeneous coordinates
x1 = x1 / x1[2]
x2 = x2 / x2[2]

# Verify the epipolar constraint
epipolar_constraint = np.dot(x2.T, np.dot(F, x1))
print("Epipolar constraint (x2^T F x1):", epipolar_constraint)

# Camera matrix P2
e2 = np.array([0, 1, 0])
e2_skew = np.array([[0, -e2[2], e2[1]], [e2[2], 0, -e2[0]], [-e2[1], e2[0], 0]])
P2 = np.hstack((e2_skew @ F, e2.reshape(3, 1)))

# Find the camera center of P2 (null space of P2)
U, S, Vh = np.linalg.svd(P2)
camera_center_P2 = Vh[-1]
camera_center_P2 = camera_center_P2 / camera_center_P2[-1]  # Normalize to homogeneous coordinates

print("Camera center of P2:", camera_center_P2)
