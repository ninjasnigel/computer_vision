import numpy as np
from scipy.linalg import null_space
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import color
from numpy.linalg import svd
from mpl_toolkits.mplot3d import Axes3D

def pflat(x):
    return x[:-1,:] / x[-1,:]

def normalize_points(points):
    mean = np.mean(points[:2], axis=1)
    std_dev = np.std(points[:2], axis=1)
    T = np.array([[1/std_dev[0], 0, -mean[0]/std_dev[0]],
                [0, 1/std_dev[1], -mean[1]/std_dev[1]],
                [0, 0, 1]])
    normalized_points = T @ points
    return normalized_points, T, mean, std_dev


def draw_line(ax, line, img_shape):
    """
    Draw a line given by 'ax + by + c = 0' on the axes 'ax'.
    """
    a, b, c = line
    x0, x1 = 0, img_shape[1]  # x-coordinates for the left and right edges of the image

    # Calculate the corresponding y-coordinates using the line equation
    # y = (-a*x - c) / b
    if b != 0:
        y0, y1 = (-a*x0 - c) / b, (-a*x1 - c) / b
        ax.plot([x0, x1], [y0, y1], linewidth=1, color='yellow')

def plot_points_lines(image, points, lines):
    """
    Plot points and their corresponding epipolar lines on the given image.
    """
    fig, ax = plt.subplots()
    ax.imshow(image)
    
    # Draw each point and its corresponding line
    for point, line in zip(points.T, lines.T):
        ax.scatter(point[0], point[1], c='blue', s=20)
        draw_line(ax, line, image.shape)

    plt.show()

def estimate_F_DLT(x1s, x2s):
    # Normalize the points
    x1s_normalized, T1, _, _ = normalize_points(x1s)
    x2s_normalized, T2, _, _ = normalize_points(x2s)

    # Set up matrix M
    num_points = x1s.shape[1]
    M = np.zeros((num_points, 9))
    for i in range(num_points):
        x1 = x1s_normalized[:, i]
        x2 = x2s_normalized[:, i]
        M[i] = [x2[0]*x1[0], x2[0]*x1[1], x2[0], x2[1]*x1[0], x2[1]*x1[1], x2[1], x1[0], x1[1], 1]

    # Solve using SVD
    U, S, Vh = svd(M)
    v = Vh[-1]  # Last row of Vh (V.T)

    # Construct fundamental matrix
    F_tilde = v.reshape(3, 3)

    # Enforce rank-2 constraint
    Uf, Sf, Vhf = svd(F_tilde)
    Sf[2] = 0  # Set the smallest singular value to zero
    F_tilde = Uf @ np.diag(Sf) @ Vhf

    return F_tilde, T1, T2

def estimate_E_DLT(x1s, x2s):
    # Normalize the points
    x1s_normalized, T1, _, _ = normalize_points(x1s)
    x2s_normalized, T2, _, _ = normalize_points(x2s)

    # Set up matrix M
    num_points = x1s.shape[1]
    M = np.zeros((num_points, 9))
    for i in range(num_points):
        x1 = x1s_normalized[:, i]
        x2 = x2s_normalized[:, i]
        M[i] = [x2[0]*x1[0], x2[0]*x1[1], x2[0], x2[1]*x1[0], x2[1]*x1[1], x2[1], x1[0], x1[1], 1]

    # Solve using SVD
    U, S, Vh = svd(M)
    v = Vh[-1]  # Last row of Vh (V.T)

    # Construct essential matrix
    E_tilde = v.reshape(3, 3)

    # Enforce the constraints on the essential matrix
    Ue, Se, Vhe = np.linalg.svd(E_tilde)
    Se = [1, 1, 0]  # Two singular values set to 1, and the third to 0
    E = Ue @ np.diag(Se) @ Vhe

    return E

def estimate_F_DLT_no_normalization(x1s, x2s):

    # Set up matrix M
    num_points = x1s.shape[1]
    M = np.zeros((num_points, 9))
    for i in range(num_points):
        x1 = x1s[:, i]
        x2 = x2s[:, i]
        M[i] = [x2[0]*x1[0], x2[0]*x1[1], x2[0], x2[1]*x1[0], x2[1]*x1[1], x2[1], x1[0], x1[1], 1]

    # Solve using SVD
    U, S, Vh = svd(M)
    v = Vh[-1]  # Last row of Vh (V.T)

    # Construct fundamental matrix
    F_tilde = v.reshape(3, 3)

    # Enforce rank-2 constraint
    Uf, Sf, Vhf = svd(F_tilde)
    Sf[2] = 0  # Set the smallest singular value to zero
    F_tilde = Uf @ np.diag(Sf) @ Vhf

    return F_tilde, T1, T2

def compute_epipolar_lines(F, x1s):
    # Compute the epipolar lines for points in x1s
    lines = F @ x1s
    return lines

def distance_point_line(point, line):
    # Compute the distance between a point and a line
    # The line is given as ax + by + c = 0, and the point is (x, y, 1)
    a, b, c = line
    x, y, _ = point
    distance = abs(a*x + b*y + c) / np.sqrt(a**2 + b**2)
    return distance

def compute_epipolar_errors(F, x1s, x2s):
    # Compute epipolar lines for x2s
    lines = compute_epipolar_lines(F, x1s)

    # Compute distances for each point-line pair
    distances = np.array([distance_point_line(x2s[:, i], lines[:, i]) for i in range(x2s.shape[1])])

    return distances

def extract_P_from_E(E):
    # Perform SVD on the essential matrix
    U, _, Vt = np.linalg.svd(E)

    # Ensure a proper rotation matrix by enforcing det(U * Vt) > 0
    if np.linalg.det(U @ Vt) < 0:
        Vt = -Vt

    # Auxiliary skew-symmetric matrix for the cross product
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    # Four possible solutions for the camera matrix
    P1 = np.hstack((U @ W @ Vt, U[:, 2].reshape(-1, 1)))
    P2 = np.hstack((U @ W @ Vt, -U[:, 2].reshape(-1, 1)))
    P3 = np.hstack((U @ W.T @ Vt, U[:, 2].reshape(-1, 1)))
    P4 = np.hstack((U @ W.T @ Vt, -U[:, 2].reshape(-1, 1)))

    return [P1, P2, P3, P4]

def triangulate_point(point1, point2, M1, M2):
    """
    Triangulate a 3D point from two corresponding 2D points in different images.
    
    :param point1: 2D point in the first image.
    :param point2: 2D point in the second image.
    :param M1: Camera matrix for the first image.
    :param M2: Camera matrix for the second image.
    :return: Triangulated 3D point.
    """
    A = np.zeros((4, 4))
    A[0] = point1[0] * M1[2] - M1[0]
    A[1] = point1[1] * M1[2] - M1[1]
    A[2] = point2[0] * M2[2] - M2[0]
    A[3] = point2[1] * M2[2] - M2[1]

    _, _, Vh = np.linalg.svd(A)
    X = Vh[-1]
    return X / X[-1]

def is_in_front_of_camera(P, X):
    # Convert to homogenous coordinates
    X_homog = np.append(X, 1)
    # Transform point into camera coordinates
    X_cam = P @ X_homog
    # Check if the z-coordinate is positive
    return X_cam[2] > 0

def count_points_in_front(P1, P2, x1s, x2s):
    count = 0
    for i in range(x1s.shape[1]):
        X = triangulate_point(P1, P2, x1s[:, i], x2s[:, i])
        if is_in_front_of_camera(P1, X) and is_in_front_of_camera(P2, X):
            count += 1
    return count

def project_points(P, points):
    projected_points = []
    for X in points:
        X_homog = np.append(X, 1)
        x_projected = P @ X_homog
        x_projected /= x_projected[2]  # Normalize to convert to homogenous coordinates
        projected_points.append(x_projected[:2])
    return np.array(projected_points).T

def camera_center_axis(P):
    from scipy.linalg import null_space

    # Calculate the camera center as the null space of P
    C = null_space(P)
    C = C / C[-1]  # Normalize to convert to homogeneous coordinates

    # The principal axis can be approximated by the third column of M normalized
    M = P[:, :-1]
    a = M[-1, :]
    print(a, a / np.linalg.norm(a), "principal axis")
    a = a / np.linalg.norm(a)  # Normalize the principal axis

    return C.flatten()[:-1], a  # Flatten and exclude the last component of C

def plot_camera(P, s=1, ax=None):

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    C, a = camera_center_axis(P)

    # Compute the end point of the principal axis scaled by s
    end_point = C + s * a

    # Plot the camera center and principal axis
    ax.scatter(C[0], C[1], C[2], c='r', marker='o')
    ax.plot([C[0], end_point[0]], [C[1], end_point[1]], [C[2], end_point[2]], 'r-')

# Load the data
mat = scipy.io.loadmat('ass3/data/compEx1data.mat')
x1s, x2s = mat['x'][0][0], mat['x'][1][0]

# Estimate the fundamental matrix
F = estimate_F_DLT(x1s, x2s)
print("Estimated Fundamental Matrix:\n", F)

# Select a pair of corresponding points
x1 = x1s[:, 0]  # First point in the first image
x2 = x2s[:, 0]  # Corresponding point in the second image

# Load your image here and select 20 random points
image1 = plt.imread('ass3/data/kronan1.JPG')
image2 = plt.imread('ass3/data/kronan2.JPG')

selected_indices = np.random.choice(x2s.shape[1], 20, replace=False)
selected_x2s = x2s[:, selected_indices]
selected_x1s = x1s[:, selected_indices]

# Compute the un-normalized fundamental matrix
F_normalized, T1, T2 = estimate_F_DLT(x1s, x2s)
F_unnormalized, _, _ = estimate_F_DLT_no_normalization(x1s, x2s)

# Check epipolar constraint
print("Epipolar constraint F unnormalized:", x2.T @ F_unnormalized @ x1)
print("Epipolar constraint F normalized:", x2.T @ F_normalized @ x1)

# Compute epipolar lines for the selected points
epipolar_lines_normalized = compute_epipolar_lines(F_normalized, selected_x1s)
epipolar_lines_unnormalized = compute_epipolar_lines(F_unnormalized, selected_x1s)

# Plot points and epipolar lines
plot_points_lines(image1, selected_x1s, epipolar_lines_normalized)
plot_points_lines(image1, selected_x1s, epipolar_lines_unnormalized)

# Compute errors
errors = compute_epipolar_errors(F_normalized, x1s, x2s)
plt.hist(errors, bins=100)
plt.title("Histogram of Epipolar Errors (Normalized)")
plt.xlabel("Distance")
plt.ylabel("Frequency")
plt.show()

# Compute mean distance
mean_distance = np.mean(errors)
print("Mean distance normalized:", mean_distance)

# Compute errors
errors = compute_epipolar_errors(F_unnormalized, x1s, x2s)
plt.hist(errors, bins=100)
plt.title("Histogram of Epipolar Errors (Unnormalized)")
plt.xlabel("Distance")
plt.ylabel("Frequency")
plt.show()

# Compute mean distance
mean_distance = np.mean(errors)
print("Mean distance unnormalized:", mean_distance)


# ---------------------- EX2 -----------------------
print('---------------------- EX2 -----------------------')
# ---------------------- EX2 -----------------------

mat = scipy.io.loadmat('ass3/data/compEx2data.mat')
K = mat['K']

x1s_normalized = np.linalg.inv(K) @ x1s
x2s_normalized = np.linalg.inv(K) @ x2s

E = estimate_E_DLT(x1s_normalized, x2s_normalized)

# Check the epipolar constraints for a set of points
epipolar_constraints = [x2s_normalized[:, i].T @ E @ x1s_normalized[:, i] for i in range(x1s_normalized.shape[1])]

mean_constraint = np.mean(epipolar_constraints)

print("Mean of epipolar constraints:", mean_constraint)

F = np.linalg.inv(K).T @ E @ np.linalg.inv(K)


# Select 20 random points from the second image
selected_indices = np.random.choice(x2s.shape[1], 20, replace=False)
selected_x2s = x2s[:, selected_indices]
selected_x1s = x1s[:, selected_indices]

# Compute epipolar lines
epipolar_lines = compute_epipolar_lines(F, selected_x1s)

# Plot the points and their corresponding epipolar lines on the image
plot_points_lines(image2, selected_x2s, epipolar_lines)

# Compute distances
errors = compute_epipolar_errors(F, selected_x1s, selected_x2s)

# ---------------------- EX3 -----------------------
print('---------------------- EX3 -----------------------')
# ---------------------- EX3 -----------------------

camera_matrices = extract_P_from_E(E)

best_solution = None
max_points_in_front = 0

# Identity matrix for the first camera
P1 = np.hstack((np.eye(3), np.array([[0], [0], [0]])))

for P2 in camera_matrices:
    count = count_points_in_front(P1, P2, x1s, x2s)
    if count > max_points_in_front:
        max_points_in_front = count
        best_solution = P2

print("Best solution:", best_solution)

P1_unnorm = K @ np.hstack((np.eye(3), np.array([[0], [0], [0]])))
P2_unnorm = K @ best_solution

# Triangulate each pair of points
triangulated_points = []
for i in range(len(matches)):
    idx1 = matches[i].queryIdx
    idx2 = matches[i].trainIdx
    point3D = triangulate_3D_point_DLT(keypoints1_homogeneous[idx1], keypoints2_homogeneous[idx2], M1, M2)
    triangulated_points.append(point3D)


plt.scatter(x2s[0, :], x2s[1, :], c='blue', label='Original Image Points', s=7)
plt.scatter(projected_points[0, :], projected_points[1, :], c='red', label='Projected Points', s=7)
plt.imshow(image2)
plt.legend()
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Original vs Projected Points')
plt.show()

errors = np.sqrt(np.sum((x2s[:2, :] - projected_points)**2, axis=0))
print("Mean projection error:", np.mean(errors))

# Camera center for the first camera
C1 = np.array([0, 0, 0])

# Camera center for the second camera (from the projection matrix)
C2 = -np.linalg.inv(P2_unnorm[:, :3]) @ P2_unnorm[:, 3]

# Principal axes (assuming the rotation matrix is the first 3x3 part of the projection matrix)
principal_axis1 = P1_unnorm[:, :3][2]
principal_axis2 = P2_unnorm[:, :3][2]

# Assuming triangulated_points is a list of 3D points
triangulated_points_np = np.array(triangulated_points)

# Set up a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D points
ax.scatter(triangulated_points_np[:, 0], triangulated_points_np[:, 1], triangulated_points_np[:, 2], c='b', marker='o', label='Triangulated 3D Points')

# Plotting the cameras
plot_camera(P1_unnorm, s=1, ax=ax)  # First camera
plot_camera(P2_unnorm, s=1, ax=ax)  # Second camera

# Setting labels and title
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.set_title('3D Points with Camera Centers and Principal Axes')
ax.legend()

plt.show()
