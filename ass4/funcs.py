import numpy as np
from scipy.linalg import null_space
import scipy.io
import matplotlib.pyplot as plt
from numpy.linalg import svd
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.linalg import null_space
import scipy.io
import matplotlib.pyplot as plt
from numpy.linalg import svd
from mpl_toolkits.mplot3d import Axes3D
import cv2

def normalize_points(points):
    mean = np.mean(points[:2], axis=1)
    std_dev = np.std(points[:2], axis=1)
    T = np.array([[1/std_dev[0], 0, -mean[0]/std_dev[0]],
                [0, 1/std_dev[1], -mean[1]/std_dev[1]],
                [0, 0, 1]])
    normalized_points = T @ points
    return normalized_points, T, mean, std_dev

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

    return F_tilde, 0, 0

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

def convert_E_to_F(E, K1, K2):
    # Convert the essential matrix back to the fundamental matrix
    F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
    return F

def compute_epipolar_lines(F, x1s):
    # Compute the epipolar lines for points in x1s
    lines = F @ x1s
    return lines

def enforce_essential(E_approx):
    # Enforce the constraints on the essential matrix
    Ue, Se, Vhe = np.linalg.svd(E_approx)
    Se = [1, 1, 0]  # Two singular values set to 1, and the third to 0
    E = Ue @ np.diag(Se) @ Vhe
    return E

def calculate_distance(point, line):
    """Calculate the distance between a point and a line."""
    x, y, w = point
    a, b, c = line
    return np.abs(a*x + b*y + c) / np.sqrt(a**2 + b**2)

def compute_rms_error(points, lines):
    """Compute the RMS error between points and their corresponding epipolar lines."""
    distances_squared = []
    for point, line in zip(points.T, lines.T):
        distance = calculate_distance(point, line)
        distances_squared.append(distance**2)
    rms_error = np.sqrt(np.mean(distances_squared))
    return rms_error

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
    dpi = 100
    height, width, _ = image.shape
    figsize = width / dpi, height / dpi

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(image)

    # Set the axis limits to the image dimensions
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)

    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Draw each point and its corresponding line
    for point, line in zip(points.T, lines.T):
        ax.scatter(point[0], point[1], c='blue', s=20)
        draw_line(ax, line, image.shape)

    plt.show()

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

def estimate_E_robust(K, x1, x2, iterations=100, threshold=1, early_stop_inliers=5000000):
    best_E = None
    max_inliers = 0
    K_inv = np.linalg.inv(K)

    for iteration in range(iterations):
        print("Iteration", iteration)
        # Randomly select 8 point correspondences
        indices = np.random.choice(x1.shape[1], 8, replace=True)
        x1_sample = x1[:, indices]
        x2_sample = x2[:, indices]

        # Convert to homogeneous coordinates if necessary
        if x1_sample.shape[0] != 3:
            x1_sample = np.vstack([x1_sample, np.ones((1, x1_sample.shape[1]))])
        if x2_sample.shape[0] != 3:
            x2_sample = np.vstack([x2_sample, np.ones((1, x2_sample.shape[1]))])

        # Normalize and estimate E using these points
        x1_normalized = K_inv @ x1_sample
        x2_normalized = K_inv @ x2_sample
        E = estimate_E_DLT(x1_normalized, x2_normalized, normalize=False)

        # Compute Fundamental Matrix F
        F = convert_E_to_F(E, K, K)

        # Compute errors and inliers
        inliers = 0
        for i in range(x1.shape[1]):
            error = compute_point_error(F, x1[:, i], x2[:, i])
            if error < threshold**2:
                inliers += 1

        # Update best model
        if inliers > max_inliers:
            best_E = E
            max_inliers = inliers

        # Early stopping if enough inliers are found
        if max_inliers >= early_stop_inliers:
            break

    return best_E, max_inliers


def compute_point_error(F, x1, x2):
    """Compute the error for a single point correspondence."""
    l1 = F @ x2
    l2 = F.T @ x1
    d1 = calculate_distance(x1, l1)**2
    d2 = calculate_distance(x2, l2)**2
    return 0.5 * (d1 + d2)

import numpy as np

def rq(a):
    """
    Perform RQ decomposition of a matrix 'a'.
    Returns 'r' (upper triangular) and 'q' (unitary matrix).
    If 'a' is not square, 'q' is extended accordingly.
    """
    m, n = a.shape
    p = np.eye(m)[:, ::-1]  # Reversing the order of columns

    q0, r0 = np.linalg.qr(np.dot(p, a[:, :m].T).dot(p))
    r = np.dot(p, r0.T).dot(p)
    q = np.dot(p, q0.T).dot(p)

    fix = np.diag(np.sign(np.diag(r)))
    r = np.dot(r, fix)
    q = np.dot(fix, q)

    if n > m:
        q = np.concatenate((q, np.dot(np.linalg.inv(r), a[:, m:n])), axis=1)

    return r, q

def triangulate_3D_point_DLT(P1, P2, x1, x2):
    """
    Triangulate a 3D point from two corresponding 2D points in different images.
    
    :param point1: 2D point in the first image.
    :param point2: 2D point in the second image.
    :param M1: Camera matrix for the first image.
    :param M2: Camera matrix for the second image.
    :return: Triangulated 3D point.
    """
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

def decompose_E(E):
    W = np.array([[0,-1,0], [1,0,0], [0,0,1]])
    U, S, Vt = np.linalg.svd(E)

    # Ensure that U and Vt have determinant 1 (proper rotation)
    if np.linalg.det(U) < 0:
        U *= -1
    if np.linalg.det(Vt) < 0:
        Vt *= -1

    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2]

    return R1, R2, t

def find_correct_pose(K, points1, points2, E):
    R1, R2, t = decompose_E(E)

    poses = [
        (R1, t),
        (R1, -t),
        (R2, t),
        (R2, -t)
    ]

    max_in_front = 0
    correct_pose = None

    for R, t in poses:
        P1 = np.hstack((R, t.reshape(-1, 1)))
        P0 = np.eye(3, 4)  # Camera matrix for the first camera

        # Project points using both camera matrices
        points_in_front = 0
        for i in range(points1.shape[1]):
            X = triangulate_3D_point_DLT(points1[:, i], points2[:, i], P0, P1)
            if X[2] > 0 and (R @ X[:3] + t)[2] > 0:
                points_in_front += 1

        if points_in_front > max_in_front:
            max_in_front = points_in_front
            correct_pose = (R, t)

    return correct_pose

def plot_3d(points_3d, P1, P2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the 3D points
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='b', marker='o', s=5)

    # Function to extract camera position from camera matrix
    def extract_camera_pos(P):
        # Camera position is given by -R^T * t
        R = P[:, :3]
        t = P[:, 3]
        return -np.linalg.inv(R).dot(t)

    # Extract and plot camera positions
    cam_pos1 = extract_camera_pos(P1)
    cam_pos2 = extract_camera_pos(P2)

    ax.scatter(cam_pos1[0], cam_pos1[1], cam_pos1[2], c='r', marker='^')
    ax.scatter(cam_pos2[0], cam_pos2[1], cam_pos2[2], c='r', marker='^')

    # Optionally draw lines representing the principal axes of the cameras
    def draw_camera(ax, cam_pos, R, scale=1.0):
        # Define a set of 3D axes (representing the camera's principal axes)
        axes = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, scale]])
        # Transform the axes according to the camera rotation
        axes_transformed = R.dot(axes.T).T
        # Draw the axes
        for p in axes_transformed:
            ax.plot([cam_pos[0], cam_pos[0]+p[0]], 
                    [cam_pos[1], cam_pos[1]+p[1]], 
                    [cam_pos[2], cam_pos[2]+p[2]], 'r-')

    draw_camera(ax, cam_pos1, P1[:, :3], scale=0.5)
    draw_camera(ax, cam_pos2, P2[:, :3], scale=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def get_camera_matrix_pairs(E):
    # Step 1: SVD Decomposition of E
    U, S, Vt = np.linalg.svd(E)

    # Ensure proper rotation matrices
    if np.linalg.det(U) < 0:
        U *= -1
    if np.linalg.det(Vt) < 0:
        Vt *= -1

    # Step 2: Define Matrix W
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    # Step 3: Possible Rotation Matrices
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt

    # Step 4: Possible Translation Vectors
    t = U[:, 2]

    # Step 5: Construct the Four Possible Camera Matrix Pairs
    camera_matrix_pairs = [
        (np.hstack((R1, t.reshape(-1, 1))), np.eye(3, 4)),  # First pair
        (np.hstack((R1, -t.reshape(-1, 1))), np.eye(3, 4)), # Second pair
        (np.hstack((R2, t.reshape(-1, 1))), np.eye(3, 4)),  # Third pair
        (np.hstack((R2, -t.reshape(-1, 1))), np.eye(3, 4))  # Fourth pair
    ]

    return camera_matrix_pairs


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

def triangulate_3D_point_DLT(point1, point2, M1, M2):
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


def is_point_in_front_of_camera(P, point):
    """
    Check if a 3D point is in front of a camera.
    P: Camera matrix
    point: 3D point in homogeneous coordinates
    Returns True if the point is in front of the camera.
    """
    transformed_point = P @ point
    # Check if the Z-coordinate is positive
    return transformed_point[2] > 0

def filter_points_in_front_of_cameras(P1, P2, points):
    """
    Filter points that are in front of both cameras.
    P1, P2: Camera matrices
    points: List or array of 3D points in homogeneous coordinates
    Returns a list of points that are in front of both cameras.
    """
    filtered_points = []
    for point in points:
        if is_point_in_front_of_camera(P1, point) and is_point_in_front_of_camera(P2, point):
            filtered_points.append(point)
    return np.array(filtered_points)

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

def count_points_in_front(P1, P2, x1s, x2s):
    count = 0
    for i in range(x1s.shape[1]):
        X = triangulate_point(P1, P2, x1s[:, i], x2s[:, i])
        if is_in_front_of_camera(P1, X) and is_in_front_of_camera(P2, X):
            count += 1
    return count

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

def compute_pixel_error(projected_points, matched_keypoints, K):
    return np.linalg.norm(projected_points - matched_keypoints, axis=1)

def project_points(P, points):
    projected_points = []
    for X in points:
        X_homog = np.append(X, 1)
        x_projected = P @ X_homog
        x_projected /= x_projected[2]  # Normalize to convert to homogenous coordinates
        projected_points.append(x_projected[:2])
    return np.array(projected_points).T

def ComputeReprojectionError(P_1, P_2, X_j, x_1j, x_2j):
    # Project the 3D point onto each camera
    x_proj_1 = P_1 @ X_j
    x_proj_2 = P_2 @ X_j

    # Normalize to get the pixel coordinates
    x_proj_1 /= x_proj_1[2]
    x_proj_2 /= x_proj_2[2]

    # Compute reprojection error
    err_1 = np.linalg.norm(x_proj_1[:2] - x_1j)
    err_2 = np.linalg.norm(x_proj_2[:2] - x_2j)
    err = err_1 + err_2

    # Compute residuals
    res = np.hstack([(x_proj_1[:2] - x_1j), (x_proj_2[:2] - x_2j)])

    return err, res

import numpy as np

def LinearizeReprojErr(P_1, P_2, X_j, x_1j, x_2j):
    # Project the 3D point onto each camera
    x_proj_1 = P_1 @ X_j
    x_proj_2 = P_2 @ X_j

    # Compute residuals
    res_1 = x_proj_1[:2] / x_proj_1[2] - x_1j
    res_2 = x_proj_2[:2] / x_proj_2[2] - x_2j
    r = np.hstack([res_1, res_2])

    # Compute Jacobian
    J = np.zeros((4, 3))  # 4 rows (2 for each camera) and 3 columns (for X, Y, Z)

    # Partial derivatives for the first camera
    J[0, :] = [1 / x_proj_1[2], 0, -x_proj_1[0] / (x_proj_1[2]**2)]
    J[1, :] = [0, 1 / x_proj_1[2], -x_proj_1[1] / (x_proj_1[2]**2)]

    # Partial derivatives for the second camera
    J[2, :] = [1 / x_proj_2[2], 0, -x_proj_2[0] / (x_proj_2[2]**2)]
    J[3, :] = [0, 1 / x_proj_2[2], -x_proj_2[1] / (x_proj_2[2]**2)]

    # Multiply Jacobian by the camera matrices
    J[0:2, :] = P_1[0:2, 0:3] - x_proj_1[0] * P_1[2:3, 0:3] / x_proj_1[2]
    J[2:4, :] = P_2[0:2, 0:3] - x_proj_2[0] * P_2[2:3, 0:3] / x_proj_2[2]

    return r, J


def ComputeUpdate(r, J, mu):
    # Compute the LM update
    A = J.T @ J + mu * np.eye(J.shape[1])
    g = J.T @ r
    delta_X_j = -np.linalg.inv(A) @ g
    return delta_X_j

def levenberg_marquardt(P1_unnorm, P2_unnorm, filtered_points_in_front, x1, x2, max_iterations=100, mu=0.01, threshold=1e-5):
    # Parameters for LM optimization
    max_iterations = 100
    convergence_threshold = 1e-6
    mu = 0.01  # Initial damping factor

    if mu < 0.02:
        return filtered_points_in_front

    for i in range(filtered_points_in_front.shape[0]):
        X_j = filtered_points_in_front[i, :]

        for iteration in range(max_iterations):
            # Compute reprojection error and residuals
            err, r = ComputeReprojectionError(P1_unnorm, P2_unnorm, X_j, x1[:, i], x2[:, i])

            # Linearize reprojection error
            r, J = LinearizeReprojErr(P1_unnorm, P2_unnorm, X_j, x1[:, i], x2[:, i])

            # Compute LM update
            delta_X_j = ComputeUpdate(r, J, mu)

            # Update the 3D point
            X_j += delta_X_j

            # Check for convergence
            if np.linalg.norm(delta_X_j) < convergence_threshold:
                break

        # Update the filtered point
        filtered_points_in_front[i, :] = X_j

    return filtered_points_in_front