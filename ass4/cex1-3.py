from funcs import *
import sys

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
E_robust, inliers = estimate_E_robust(K, x1s, x2s, iterations=150)

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

def pipeline_to_3D(K, img1, img2, use_levenberg_marquardt=False):
    
    sift = cv2.xfeatures2d.SIFT_create()

    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    print('what')

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    print('what2')

    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key = lambda x:x.distance)

    x1 = np.array([keypoints1[m.queryIdx].pt for m in matches]).T
    x2 = np.array([keypoints2[m.trainIdx].pt for m in matches]).T

    # Ensure points are in homogeneous coordinates
    x1 = np.vstack([x1, np.ones((1, x1.shape[1]))])
    x2 = np.vstack([x2, np.ones((1, x2.shape[1]))])

    # Estimate the essential matrix
    print('before robust')
    E, inliers = estimate_E_robust(K, x1, x2, iterations=100, threshold=2)
    print('after robust')
    #E = estimate_E_DLT(x1, x2)
    print('inliers:', inliers)

    print('Estimated E with RANSAC:', E)

    camera_matrices = extract_P_from_E(E)

    best_solution = None
    max_points_in_front = 0

    # Identity matrix for the first camera
    P1 = np.hstack((np.eye(3), np.array([[0], [0], [0]])))

    for P2 in camera_matrices:
        count = count_points_in_front(P1, P2, x1, x2)
        if count > max_points_in_front:
            max_points_in_front = count
            best_solution = P2

    print("Best solution:", best_solution)

    P1_unnorm = K @ np.hstack((np.eye(3), np.array([[0], [0], [0]])))
    P2_unnorm = K @ best_solution

    triangulated_points = [triangulate_point(P1_unnorm, P2_unnorm, x1[:, i], x2[:, i]) for i in range(x1.shape[1])]
    projected_points = project_points(P2_unnorm, triangulated_points)

    best_P =  best_solution
    best_points_3d = np.array(triangulated_points)

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
        count = count_points_in_front(P1, P2, x1, x2)
        if count > max_points_in_front:
            max_points_in_front = count
            best_solution = P2

    print("Best solution:", best_solution)

    P1_unnorm = K @ np.hstack((np.eye(3), np.array([[0], [0], [0]])))
    P2_unnorm = K @ best_solution

    projected_points = project_points(P2_unnorm, triangulated_points)

    plt.scatter(x2[0, :], x2[1, :], c='blue', label='Original Image Points')
    plt.scatter(projected_points[0, :], projected_points[1, :], c='red', label='Projected Points')
    plt.legend()
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Original vs Projected Points')
    plt.imshow(img2)
    plt.show()

    x_min, x_max = -10, 10
    y_min, y_max = -10, 10
    z_min, z_max = -20, 20

    homogeneous_points = np.hstack([best_points_3d, np.ones((best_points_3d.shape[0], 1))])

    filtered_points_in_front = filter_points_in_front_of_cameras(P1_unnorm, P2_unnorm, homogeneous_points)

    if use_levenberg_marquardt:
        print('Using Levenberg-Marquardt')
        filtered_points_in_front = levenberg_marquardt(P1_unnorm, P2_unnorm, filtered_points_in_front)

    # Plot 3d figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(filtered_points_in_front[:, 0], filtered_points_in_front[:, 1], filtered_points_in_front[:, 2], s=20, c='red', label='Filtered Points')
    #include the cameras in the plot
    plot_camera(P1_unnorm, ax=ax)
    plot_camera(P2_unnorm, ax=ax)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

    print('-----FUNC DONE-----')


# Load data
data = scipy.io.loadmat('data/kronan.mat')
K = data['K']
print("Data contents" , data.keys())

img1 = cv2.imread('data/kronan1.JPG')
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

img2 = cv2.imread('data/kronan2.JPG')
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

pipeline_to_3D(K, img1, img2)

# ---------------------- EX3 -----------------------
print('---------------------- EX3 -----------------------')
# ---------------------- EX3 -----------------------

pipeline_to_3D(K, img1, img2, use_levenberg_marquardt=True)