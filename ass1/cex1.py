import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def pflat(x):
    return x[:-1,:] / x[-1,:]

def plot_points_2D(x):
    plt.plot(x[0, :], x[1, :], '.')
    plt.axis('equal')

def plot_points_3D(ax, x):
    ax.scatter(x[0, :], x[1, :], x[2, :], marker='.')
    ax.set_box_aspect([1,1,1])

def main():
    # Load data from data/compEx1.mat
    mat = scipy.io.loadmat('data/compEx1.mat')

    # Extract the points from the data
    x2D, x3D = mat['x2D'], mat['x3D']

    # Normalize the points
    x2D_normalized, x3D_normalized = pflat(x2D), pflat(x3D)

    # Create a figure for subplots
    plt.figure(figsize=(14, 7))

    # Plot for 2D points
    plt.subplot(1, 2, 1)
    plot_points_2D(x2D_normalized)
    plt.title('Normalized 2D points')

    # Plot for 3D points
    ax = plt.subplot(1, 2, 2, projection='3d')
    plot_points_3D(ax, x3D_normalized)
    plt.title('Normalized 3D points')

    # Show combined plot
    plt.show()

if __name__ == '__main__':
    main()