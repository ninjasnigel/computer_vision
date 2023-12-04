import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from rq import rq

# Function to normalize points
def normalize_points(points):
    mean = np.mean(points[:2], axis=1)
    std_dev = np.std(points[:2], axis=1)
    T = np.array([[1/std_dev[0], 0, -mean[0]/std_dev[0]],
                  [0, 1/std_dev[1], -mean[1]/std_dev[1]],
                  [0, 0, 1]])
    normalized_points = T @ points
    return normalized_points, T, mean, std_dev

def estimate_camera_DLT(Xmodel, x):
    # Number of points
    n = Xmodel.shape[1]

    # Construct the DLT matrix
    A = []
    for i in range(n):
        Xi = Xmodel[:, i]
        xi = x[:, i]
        A.append(np.kron(Xi, [0, 0, 0, 1]) - np.kron([0, 0, 0, 1], xi))
        A.append(np.kron([1, 0, 0, 0], xi) - np.kron(Xi, [1, 0, 0, 0]))
    
    A = np.array(A)

    # Solve using SVD
    U, S, Vh = np.linalg.svd(A)
    M = Vh[-1].reshape(3, 4)

    return M

mat = scipy.io.loadmat('data/compEx3data.mat')
cube1, cube2 = plt.imread('data/cube1.JPG'), plt.imread('data/cube2.JPG')
Xmodel = mat['Xmodel']
x = mat['x']
startind = mat['startind']
endind = mat['endind']

x_image1 = x[0][0] 
x_image2 = x[0][1] 

# Then call the normalization function
normalized_x1, T1, mean1, std1 = normalize_points(x_image1)
normalized_x2, T2, mean2, std2 = normalize_points(x_image2)

# Apply DLT for each view
M1 = estimate_camera_DLT(Xmodel, normalized_x1)
M2 = estimate_camera_DLT(Xmodel, normalized_x2)

# Check smallest singular value
_, S1, _ = np.linalg.svd(M1)
_, S2, _ = np.linalg.svd(M2)

Xmodel_homogeneous = np.vstack([Xmodel, np.ones(Xmodel.shape[1])])

v1 = np.linalg.norm(M1 @ Xmodel_homogeneous)
v2 = np.linalg.norm(M2 @ Xmodel_homogeneous)

print("||M v|| for view 1:", v1)
print("||M v|| for view 2:", v2)

# Assuming x_image1 and x_image2 are the original, unnormalized points
plt.figure()
plt.imshow(cube1)
plt.plot(x_image1[0], x_image1[1], 'r.')  # Plot original points on the first image
plt.figure()
plt.imshow(cube2)
plt.plot(x_image2[0], x_image2[1], 'r.')  # Plot original points on the second image

# Plotting 3D model points
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(Xmodel[0], Xmodel[1], Xmodel[2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

K, R = rq(M1[:, :3])
print("Inner parameters of the first camera:")

print("K:", K)
print("R:", R)

print("Mean and standard deviation for first camera:")
print("Mean:", mean1)
print("Standard deviation:", std1)

print("Calibration matrix of the first camera:")
print(K / K[2, 2])

print("Mean and standard deviation for second camera:")
print("Mean:", mean2)
print("Standard deviation:", std2)
