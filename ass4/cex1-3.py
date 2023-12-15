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

if "a" in sys.argv:
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

if "a" in sys.argv:
    # Plot the points and their corresponding epipolar lines in both images
    plot_points_lines(img2, selected_x2s, lines2[:, selected_indices])

# Estimate E using RANSAC
E_robust, inliers = estimate_E_robust(K, x1s, x2s, iterations=1)

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

if "a" in sys.argv:
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

if "a" in sys.argv:
    # Plot the points and their corresponding epipolar lines in both images
    plot_points_lines(img2, selected_x2s, lines2[:, selected_indices])

# ---------------------- EX2 -----------------------
print('---------------------- EX2 -----------------------')
# ---------------------- EX2 -----------------------

def pipeline_to_3D(K, img1, img2):
    
    sift = cv2.xfeatures2d.SIFT_create()

    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key = lambda x:x.distance)

    # Convert matches to points
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).T
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).T

    # Convert to homogeneous coordinates (add a row of 1s)
    keypoints1_homogeneous = np.array([kp.pt + (1,) for kp in keypoints1])
    keypoints2_homogeneous = np.array([kp.pt + (1,) for kp in keypoints2])

    K_inv = np.linalg.inv(K)

    x1 = points1.reshape(2, -1)
    x2 = points2.reshape(2, -1)

    x1_homogeneous = np.vstack([x1, np.ones((1, x1.shape[1]))])
    x2_homogeneous = np.vstack([x2, np.ones((1, x2.shape[1]))])

    # Estimate E using RANSAC with homogeneous coordinates
    E, inliers = estimate_E_robust(K, x1_homogeneous, x2_homogeneous, iterations=250, threshold=15)

    print('Estimated E with RANSAC:', E)

    print('Inliers:', inliers)
    triangulated_points = []
    for i, (M1, M2) in enumerate(get_camera_matrix_pairs(E)):
        
        if "ass3" in sys.argv:
            best_solution = M2

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
            plt.imshow(img2)
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

        if "ass2" in sys.argv:
            # TESTING 2D-------------------
            triangulated_points = []
            for i in range(len(matches)):
                idx1 = matches[i].queryIdx
                idx2 = matches[i].trainIdx
                point3D = triangulate_3D_point_DLT(keypoints1_homogeneous[idx1], keypoints2_homogeneous[idx2], M1, M2)
                triangulated_points.append(point3D)

            triangulated_points = np.array(triangulated_points)

            # Project the points
            projected_points1 = project_points(triangulated_points, M1)
            projected_points2 = project_points(triangulated_points, M2)

            # Assuming img1 and img2 are your image arrays, get their dimensions
            height1, width1 = img1.shape[:2]
            height2, width2 = img2.shape[:2]

            # Function to filter points based on image dimensions
            def filter_points(points, width, height):
                return [p for p in points if 0 <= p[0] < width and 0 <= p[1] < height]

            # Filter the projected points
            filtered_projected_points1 = filter_points(projected_points1, width1, height1)
            filtered_projected_points2 = filter_points(projected_points2, width2, height2)

            # Plotting for the first image with filtered points
            plt.figure(figsize=(10, 8))
            plt.imshow(img1, cmap='gray')
            plt.scatter([kp.pt[0] for kp in keypoints1], [kp.pt[1] for kp in keypoints1], c='r', s=5)
            plt.scatter([p[0] for p in filtered_projected_points1], [p[1] for p in filtered_projected_points1], c='b', s=5)
            plt.title('Image 1: Original SIFT Points (Red) and Projected Points (Blue)')
            plt.show()

            # TESTING 3D-------------------

            camera_center1 = -np.linalg.inv(M1[:, :3]) @ M1[:, 3]
            camera_center2 = -np.linalg.inv(M2[:, :3]) @ M2[:, 3]

            # extract the principal axis
            C1, a1 = camera_center_axis(M1)
            C2, a2 = camera_center_axis(M2)

            # Create arrays of matched keypoints
            matched_keypoints1 = np.array([keypoints1[match.queryIdx].pt for match in matches])
            matched_keypoints2 = np.array([keypoints2[match.trainIdx].pt for match in matches])

            errors1 = compute_pixel_error(projected_points1, matched_keypoints1, K)
            errors2 = compute_pixel_error(projected_points2, matched_keypoints2, K)

            # Rest of your filtering and plotting logic
            valid_indices = (errors1 < 10) & (errors2 < 10)
            filtered_3D_points = triangulated_points[valid_indices]

            x_limit = (-50, 50)
            y_limit = (-50, 50)
            z_limit = (-50, 50)

            # Apply the limits
            filtered_3D_points = filtered_3D_points[(filtered_3D_points[:, 0] > x_limit[0]) & (filtered_3D_points[:, 0] < x_limit[1]) &
                                                    (filtered_3D_points[:, 1] > y_limit[0]) & (filtered_3D_points[:, 1] < y_limit[1]) &
                                                    (filtered_3D_points[:, 2] > z_limit[0]) & (filtered_3D_points[:, 2] < z_limit[1])]


            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Plot the filtered triangulated 3D points
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



    # Initialize variables to hold the best result
    best_solution = None
    max_inliers = 0

    # Normalize camera matrix pairs
    """
    for i, (P1, P2) in enumerate(camera_matrix_pairs):
        # Triangulate 3D points for this camera pair
        P1_unnorm = K @ P1
        P2_unnorm = K @ P2
        points_3d = [triangulate_point(P1_unnorm, P2_unnorm, p1, p2) for p1, p2 in zip(points1, points2)]
        points_3d = np.array(points_3d)

        if points_3d.shape[1] == 3:
            points_3d = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])

        # Filter points to find the number of inliers (points in front of both cameras)
        inliers = sum((point[2] > 0 and (P2 @ point)[2] > 0) for point in points_3d)

        # Update the best solution if this one is better
        if inliers > max_inliers:
            max_inliers = inliers
            best_solution = (P1, P2, points_3d)

        # Plotting (optional for each iteration)
        plot_3d(points_3d, P1, P2)

    # Plot the best solution
    print(best_solution[2])
    if best_solution:
        P1, P2, points_3d = best_solution
    else:
        print('No solution found')
"""
    print("----func done----")

# Load data
data = scipy.io.loadmat('data/compEx2data.mat')
K = data['K']
print("Data contents" , data.keys())

img1 = cv2.imread('data/fountain1.png')
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

img2 = cv2.imread('data/fountain2.png')
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

pipeline_to_3D(K, img1, img2)