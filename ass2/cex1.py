import numpy as np
import matplotlib.pyplot as plt
from plotcams import plotcams
from rq import rq
from scipy.io import loadmat
import os

def plot_3d_points_and_images(X, Ps, imfiles, equal_axes=True):
    fig = plt.figure(figsize=(10, 5))  # Adjust the figure size as needed
    ax_points = fig.add_subplot(121, projection='3d')  # Create subplot for 3D points
    ax_image = fig.add_subplot(122)  # Create subplot for image

    # Plot 3D points
    ax_points.scatter(X[0, :], X[1, :], X[2, :], c='b', marker='.')
    plotcams(Ps, ax_points)  # Integrated plotcams
    if equal_axes:
        ax_points.set_aspect('auto')

    # Plot image
    image_path = imfiles[0][1][0]
    image = plt.imread(image_path)
    ax_image.imshow(image)
    ax_image.axis('off')

    plt.show()

def load_data(file_path):
    data = loadmat(file_path)
    X = data['X']
    P_raw = data['P']

    # P is a cell array in MATLAB, we need to extract each element (this sucked to figure out)
    Ps = []
    for i in range(P_raw.shape[1]):
        P_matrix = P_raw[0, i]
        Ps.append(P_matrix.reshape(3, 4))

    x = data['x']
    imfiles = data['imfiles']

    return X, Ps, x, imfiles

def apply_transformation(X, T):
    return T @ X

def transform_camera_matrices(Ps, T):
    return [P @ np.linalg.inv(T) for P in Ps]

def project_and_plot(P, Xs, image_path):
    x_proj = P @ Xs
    x_proj /= x_proj[2, :]  # Normalize
    plt.imshow(plt.imread(image_path))
    plt.scatter(x_proj[0, :], x_proj[1, :], c='r', marker='.', s=3)
    plt.show()

def main():
    # Load data
    X, Ps, x, imfiles = load_data('data/compEx1data.mat')

    # Plot initial 3D points and cameras
    plot_3d_points_and_images(X, Ps, imfiles)

    # Define transformations
    T1 = np.array([[1, 0, 0, 0], [0, 3, 0, 0], [0, 0, 1, 0], [1/8, 1/8, 0, 1]])
    T2 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1/16, 1/16, 0, 1]])

    # Apply transformations and plot
    for T in (T1, T2):
        X_transformed = apply_transformation(X, T)
        Ps_transformed = transform_camera_matrices(Ps, T)
        project_and_plot(Ps_transformed[0], X_transformed, "DSC_0025.jpg")

if __name__ == '__main__':
    main()
