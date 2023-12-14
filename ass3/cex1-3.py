import numpy as np
from scipy.linalg import null_space
import scipy.io
import matplotlib.pyplot as plt
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

def estimate_E_DLT(x1s, x2s, normalize=True):
    # Normalize the points
    if normalize:
        x1s_normalized, T1, _, _ = normalize_points(x1s)
        x2s_normalized, T2, _, _ = normalize_points(x2s)
    else:
        x1s_normalized = x1s
        x2s_normalized = x2s

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

def triangulate_point(P1, P2, x1, x2):
    # Construct matrix A for DLT
    A = np.array([
        x1[0] * P1[2] - P1[0],
        x1[1] * P1[2] - P1[1],
        x2[0] * P2[2] - P2[0],
        x2[1] * P2[2] - P2[1]
    ])

    # Perform SVD
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X = X / X[3]  # Dehomogenize

    return X[:3]


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

    # Invert the z-coordinate
    C[2] *= -1
    a[2] *= -1

    # Compute the end point of the principal axis scaled by s
    end_point = C + s * a

    # Plot the camera center and principal axis
    ax.scatter(C[0], C[1], C[2], c='r', marker='o')
    ax.plot([C[0], end_point[0]], [C[1], end_point[1]], [C[2], end_point[2]], 'r-')

def convert_E_to_F(E, K1, K2):
    # Convert the essential matrix back to the fundamental matrix
    F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
    return F

# Load the data
mat = scipy.io.loadmat('ass3/data/compEx1data.mat')
x1s, x2s = mat['x'][0][0], mat['x'][1][0]

# Estimate the fundamental matrix
F = estimate_F_DLT(x1s, x2s)

# Load your image here and select 20 random points
image1 = plt.imread('ass3/data/kronan1.JPG')
image2 = plt.imread('ass3/data/kronan2.JPG')

selected_indices = np.random.choice(x2s.shape[1], 20, replace=False)
selected_x2s = x2s[:, selected_indices]
selected_x1s = x1s[:, selected_indices]

# Compute the un-normalized fundamental matrix
F_normalized, T1, T2 = estimate_F_DLT(x1s, x2s)
F_unnormalized, _, _ = estimate_F_DLT_no_normalization(x1s, x2s)
F_unnormalized /= F_unnormalized[2, 2]
print("Original Estimated Fundamental Matrix:\n", F_unnormalized)

# Check epipolar constraint
epipolar_constraints = [x2s[:, i].T @ F_unnormalized @ x1s[:, i] for i in range(x1s.shape[1])]
print("Mean of epipolar constraints unnormalized:", np.mean(epipolar_constraints))

# Compute epipolar lines for the selected points
epipolar_lines_normalized = compute_epipolar_lines(F_normalized, selected_x1s)
epipolar_lines_unnormalized = compute_epipolar_lines(F_unnormalized, selected_x1s)

# Plot the points and their corresponding epipolar lines on the image
plot_points_lines(image1, selected_x1s, epipolar_lines_unnormalized)

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

E = estimate_E_DLT(x1s_normalized, x2s_normalized, normalize=False)

# Check the epipolar constraints for a set of points
epipolar_constraints = [x2s_normalized[:, i].T @ E @ x1s_normalized[:, i] for i in range(x1s_normalized.shape[1])]

mean_constraint = np.mean(epipolar_constraints)

print("Mean of epipolar constraints:", mean_constraint)

U, S, Vt = np.linalg.svd(E)
S_normalized = [1, 1, 0]  # Adjust the singular values
E_normalized = U @ np.diag(S_normalized) @ Vt

print(E_normalized)

# Convert the essential matrix back to the fundamental matrix
F = np.linalg.inv(K).T @ E_normalized @ np.linalg.inv(K)

# Select 20 random points from the second image
selected_indices = np.random.choice(x2s.shape[1], 20, replace=False)
selected_x2s = x2s[:, selected_indices]
selected_x1s = x1s[:, selected_indices]

# Compute epipolar lines
epipolar_lines = compute_epipolar_lines(F, selected_x1s)
plot_points_lines(image2, selected_x2s, epipolar_lines)
errors = compute_epipolar_errors(F, selected_x1s, selected_x2s)

average_distance = np.mean(errors)
print("Average point-to-line distance in pixels:", average_distance)

# Compute distances
errors = compute_epipolar_errors(F, selected_x1s, selected_x2s)
plt.hist(errors, bins=100)
plt.title("Histogram of Epipolar Errors")
plt.xlabel("Distance")
plt.ylabel("Frequency")
plt.show()

# ---------------------- EX3 -----------------------
print('---------------------- EX3 -----------------------')
# ---------------------- EX3 -----------------------

E = estimate_E_DLT(x1s, x2s)
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

triangulated_points = [triangulate_point(P1_unnorm, P2_unnorm, x1s[:, i], x2s[:, i]) for i in range(x1s.shape[1])]
projected_points = project_points(P2_unnorm, triangulated_points)

best_P =  best_solution
best_points_3d = np.array(triangulated_points)
# Extract the rotational part and the translation vector from the best camera matrix
R = best_P[:, :3]
t = best_P[:, 3]

# Apply the inverse transformation to the rotational part
R_unnormalized = np.linalg.inv(T2).dot(R).dot(T1)

# For the translation vector, you can directly multiply it by the inverse of T2
# since it's a 3x1 vector
t_unnormalized = np.linalg.inv(T2).dot(t)

# Reconstruct the unnormalized camera matrix
best_P_unnormalized = np.hstack((R_unnormalized, t_unnormalized.reshape(-1, 1)))

# Calculate the mean position of the points
mean_position = np.mean(best_points_3d[:, :3], axis=0)

# Calculate the distance of each point from the mean position
distances = np.linalg.norm(best_points_3d[:, :3] - mean_position, axis=1)

# Set a threshold for filtering out outliers
threshold = distances.mean() + 2 * distances.std()

# Filter points
filtered_points = best_points_3d[distances < threshold]

# Project the 3D points
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

projected_points = project_points(P2_unnorm, triangulated_points)

plt.scatter(x2s[0, :], x2s[1, :], c='blue', label='Original Image Points')
plt.scatter(projected_points[0, :], projected_points[1, :], c='red', label='Projected Points')
plt.legend()
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Original vs Projected Points')
plt.imshow(image2)
plt.show()

# Plot 3d figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(filtered_points[:, 0], filtered_points[:, 1], filtered_points[:, 2], s=20, c='red', label='Filtered Points')
#include the cameras in the plot
plot_camera(P1_unnorm, ax=ax)
plot_camera(P2_unnorm, ax=ax)
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(-10, -25)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()