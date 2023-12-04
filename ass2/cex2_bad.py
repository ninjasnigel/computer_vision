import numpy as np
import matplotlib.pyplot as plt
from plotcams import plotcams
from rq import rq
from scipy.io import loadmat
#need to import svd
from numpy.linalg import svd

#Load files
data = loadmat('data/compEx3data.mat')

x = data['x']  # Replace 'x' with the actual key for the measured projections in the .mat file
Xmodel = data['Xmodel']  # Replace 'Xmodel' with the actual key for the model points

# Function to normalize points
def normalize_points(points):
    mean_x = np.mean(points[0, :])
    mean_y = np.mean(points[1, :])
    std_x = np.std(points[0, :])
    std_y = np.std(points[1, :])

    N = np.array([[1/std_x, 0, -mean_x/std_x],
                  [0, 1/std_y, -mean_y/std_y],
                  [0, 0, 1]])
    
    normalized_points = N @ np.vstack([points, np.ones((1, points.shape[1]))])
    return normalized_points, N

def estimate_camera_DLT(x, X):
    """
    Estimate camera projection matrix using Direct Linear Transform (DLT).

    :param x: Normalized 2D points in the image (homogeneous coordinates).
    :param X: 3D points in the model (homogeneous coordinates).
    :return: Estimated camera matrix.
    """
    num_points = x.shape[1]
    A = []  # System of equations

    for i in range(num_points):
        X_i = X[:, i]
        x_i = x[:, i]
        A.append(np.kron(X_i, [1, 0, 0, 0]) * -x_i[2])  # Corresponds to x-coordinate
        A.append(np.kron(X_i, [0, 1, 0, 0]) * -x_i[2])  # Corresponds to y-coordinate
        A.append(np.kron(X_i, [0, 0, 1, 0]) * -x_i[0])
        A.append(np.kron(X_i, [0, 0, 0, 1]) * -x_i[1])

    A = np.array(A)  # Convert list to numpy array

    # Solving using SVD
    U, S, Vh = svd(A)
    M = Vh[-1, :].reshape(3, 4)  # The last row of Vh (transposed V) reshaped into a 3x4 matrix

    return M


# Normalize the points
normalized_points, N = normalize_points(x)

# Plotting normalized points
plt.scatter(normalized_points[0, :], normalized_points[1, :])
plt.title('Normalized Points')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.show()
