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
        print('Iteration', iteration)
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

# ---------------------- EX1 -----------------------
print('---------------------- EX1 -----------------------')
# ---------------------- EX1 -----------------------

# Load data
data = scipy.io.loadmat('data/compEx1data.mat')
K = data['K']
x = data['x']
x1s = x[0][0]
x2s = x[0][1]

# Load images from data (round_church1.jpg and round_church2.jpg)
img1 = plt.imread('data/round_church1.jpg')
img2 = plt.imread('data/round_church2.jpg')

# Solve for the essential matrix E with an eight point algorithm using all the point correspon-
# dences. Remember to normalize image points with K beforehand.
x1s_normalized = np.linalg.inv(K) @ x1s
x2s_normalized = np.linalg.inv(K) @ x2s

E = estimate_E_DLT(x1s_normalized, x2s_normalized, normalize=False)
print('E = \n', E)

"""Compute the epipolar lines l2 = F x1 and l1 = F T x2."""
# Compute the fundamental matrix F
F = convert_E_to_F(E, K, K)
print('F = \n', F)

# Compute the epipolar lines
lines1 = compute_epipolar_lines(F, x1s)
lines2 = compute_epipolar_lines(F.T, x2s)

# Compute the RMS distance between the points and the epipolar lines
rms_error_x1_l1 = compute_rms_error(x1s, lines1)
rms_error_x2_l2 = compute_rms_error(x2s, lines2)

print('RMS Error for x1 and l1:', rms_error_x1_l1)
print('RMS Error for x2 and l2:', rms_error_x2_l2)

# Compute epipolar errors
errors = compute_epipolar_errors(F, x1s, x2s)

# Plot histogram of epipolar errors
plt.hist(errors, bins=100)
plt.title('Histogram of epipolar errors')
plt.xlabel('Distance')
plt.ylabel('Number of points')
plt.show()

# Plot the points and their corresponding epipolar lines in both images
selected_indices = np.random.choice(x2s.shape[1], 20, replace=False)
selected_x2s = x2s[:, selected_indices]
selected_x1s = x1s[:, selected_indices]

# Plot the points and their corresponding epipolar lines in both images
plot_points_lines(img2, selected_x2s, lines2[:, selected_indices])

# Estimate E using RANSAC
E_robust, inliers = estimate_E_robust(K, x1s, x2s, iterations=100)

print('Estimated E with RANSAC:', E_robust)
print('Number of inliers:', inliers)

# Compute the fundamental matrix F
F_robust = convert_E_to_F(E_robust, K, K)
print('F_robust = \n', F_robust)

# Compute epipolar lines
lines1 = compute_epipolar_lines(F_robust, x1s)
lines2 = compute_epipolar_lines(F_robust.T, x2s)

# Compute the RMS distance between the points and the epipolar lines
rms_error_x1_l1 = compute_rms_error(x1s, lines1)
rms_error_x2_l2 = compute_rms_error(x2s, lines2)

print('RMS Error for x1 and l1:', rms_error_x1_l1)
print('RMS Error for x2 and l2:', rms_error_x2_l2)

# Compute epipolar errors
errors = compute_epipolar_errors(F_robust, x1s, x2s)

# Plot histogram of epipolar errors
plt.hist(errors, bins=100)
plt.title('Histogram of epipolar errors')
plt.xlabel('Distance')
plt.ylabel('Number of points')
plt.show()

# Plot the points and their corresponding epipolar lines in both images
selected_indices = np.random.choice(x2s.shape[1], 20, replace=False)
selected_x2s = x2s[:, selected_indices]
selected_x1s = x1s[:, selected_indices]

# Plot the points and their corresponding epipolar lines in both images
plot_points_lines(img2, selected_x2s, lines2[:, selected_indices])

# ---------------------- EX2 -----------------------
print('---------------------- EX2 -----------------------')
# ---------------------- EX2 -----------------------

# Load data
data = scipy.io.loadmat('data/compEx2data.mat')
K = data['K']

img1 = cv2.imread('data/fountain1.png')
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

img2 = cv2.imread('data/fountain2.png')
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()

keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
print('Number of features in image 1:', len(keypoints1))
print('Number of features in image 2:', len(keypoints2))

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

matches = bf.match(descriptors1, descriptors2)

matches = sorted(matches, key = lambda x:x.distance)
print('Number of matches:', len(matches))

matched_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:50], None, flags=2)

#set size of window smaller
cv2.namedWindow('Matched Features', cv2.WINDOW_NORMAL)
cv2.imshow('Matched Features', matched_img)
cv2.imwrite('matched_features.jpg', matched_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""Now you should find the essential matrix describing the transformation between the two images.
Because not all matches are correct, you need to use RANSAC to find a set of good correspondences
(inliers). To estimate the essential matrix use the function estimate_E_robust(K,x1,x2) that you
created in the previous computer exercise.
How many inliers did you find?"""

# Estimate E using RANSAC
E_robust, inliers = estimate_E_robust(K, x1s, x2s, iterations=100)

# Evaluate the Result
print('Estimated E with RANSAC:', E_robust)
print('Number of inliers:', np.sum(inliers))

"""After getting the robust essential matrix estimation, you should find the camera matrix of the second
view. Remember that there are 4 possible solutions (see Lecture 6 or the Theoretical Exercise 7 of
HA3)! You should pick the solution that has more points in front of the camera.

Find the 4 possible camera matrices pairs for the essential matrix that you estimated, and for
each them:
(a) Triangulate the 3D points using the camera matrix pair and image points;
(b) Get the camera centers and principal direction of both cameras;
(c) Plot everything in 3D;"""

# Create arrays of matched keypoints

K_inv = np.linalg.inv(K)

keypoints1_normalized = np.array([K @ np.array(kp.pt + (1,)).T for kp in keypoints1])
keypoints2_normalized = np.array([K @ np.array(kp.pt + (1,)).T for kp in keypoints2])

M1_normalized = K @ x1s
M2_normalized = K @ x2s

triangulated_points = []
for i in range(len(matches)):
    idx1 = matches[i].queryIdx
    idx2 = matches[i].trainIdx
    point3D = triangulate_3D_point_DLT(keypoints1_normalized[idx1][:2], keypoints2_normalized[idx2][:2], M1_normalized, M2_normalized)
    triangulated_points.append(point3D)

matched_keypoints1 = np.array([keypoints1[match.queryIdx].pt for match in matches])
matched_keypoints2 = np.array([keypoints2[match.trainIdx].pt for match in matches])

x_limit = (-500, 500)
y_limit = (-500, 500)
z_limit = (-500, 500)

filtered_3D_points = np.array(triangulated_points)

# Apply the limits
filtered_3D_points = filtered_3D_points[(filtered_3D_points[:, 0] > x_limit[0]) & (filtered_3D_points[:, 0] < x_limit[1]) &
                                           (filtered_3D_points[:, 1] > y_limit[0]) & (filtered_3D_points[:, 1] < y_limit[1]) &
                                           (filtered_3D_points[:, 2] > z_limit[0]) & (filtered_3D_points[:, 2] < z_limit[1])]

# Get the camera centers and principal directions
camera_center1 = -np.linalg.inv(x1s[:, :3]) @ x1s[:, 3]
camera_center2 = -np.linalg.inv(x2s[:, :3]) @ x2s[:, 3]

# extract the principal axis
C1, a1 = camera_center_axis(x1s)
C2, a2 = camera_center_axis(x1s)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(filtered_3D_points[:, 0], filtered_3D_points[:, 1], filtered_3D_points[:, 2], c='blue', marker='o', label='Triangulated Points')

# Plot the camera positions
ax.scatter(camera_center1[0], camera_center1[1], camera_center1[2], c='red', marker='^', label='Camera 1')
ax.scatter(camera_center2[0], camera_center2[1], camera_center2[2], c='green', marker='^', label='Camera 2')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Visualization with Cameras and Cube Model')
ax.legend()

plt.show()