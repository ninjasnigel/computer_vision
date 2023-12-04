from numpy.linalg import svd
import matplotlib.pyplot as plt
import numpy as np

def plotcams(P, ax):
    num_cameras = len(P)
    c = np.zeros((4, num_cameras))
    v = np.zeros((3, num_cameras))

    for i in range(num_cameras):
        # Computing the null space of P[i]
        U, S, Vh = svd(P[i])
        null_space = Vh[-1, :]
        c[:, i] = null_space / null_space[3]

        # Extracting the third row for the direction vectors
        v[:, i] = P[i][2, :3]

    # Plotting the camera positions and orientations on the given Axes object
    ax.quiver(c[0, :], c[1, :], c[2, :], v[0, :], v[1, :], v[2, :], color='r', linewidth=1.5)