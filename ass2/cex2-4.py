import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from rq import rq
import cv2

# -----------------------------------------------   
# Computer exercise 2

mat = scipy.io.loadmat('ass2/data/compEx3data.mat')
cube1, cube2 = plt.imread('ass2/data/cube1.JPG'), plt.imread('ass2/data/cube2.JPG')
Xmodel = mat['Xmodel']
x = mat['x']
startind = mat['startind']
endind = mat['endind']

x_image1 = x[0][0] 
x_image2 = x[0][1]

# Normalize the points
def normalize_points(points):
    # Separate x and y coordinates
    x_coords = points[0, :]
    y_coords = points[1, :]

    # Compute mean for x and y
    mean_x = np.mean(x_coords)
    mean_y = np.mean(y_coords)

    # Subtract mean
    x_coords -= mean_x
    y_coords -= mean_y

    # Compute standard deviation for x and y
    std_x = np.std(x_coords)
    std_y = np.std(y_coords)

    # Divide by standard deviation
    x_coords /= std_x
    y_coords /= std_y

    return np.array([x_coords, y_coords]), std_x, std_y

def estimate_camera_DLT(X, x):
    # Number of points
    n = X.shape[1]

    # Convert 3D points to homogeneous coordinates
    X_homogeneous = np.vstack((X, np.ones((1, n))))

    # Construct matrix A
    A = np.zeros((2 * n, 12))
    for i in range(n):
        X_i = X_homogeneous[:, i]
        x_i = x[:, i]
        A[2*i, 4:8] = -X_i
        A[2*i, 8:] = x_i[1] * X_i
        A[2*i+1, :4] = X_i
        A[2*i+1, 8:] = -x_i[0] * X_i

    # Perform SVD
    U, S, VT = np.linalg.svd(A)

    # The camera matrix M is the last column of V (or VT.T)
    M = VT[-1, :].reshape(3, 4)

    # Check the smallest singular value
    smallest_singular_value = S[-1]

    return M, smallest_singular_value

def project_points(M, X):
    # Convert X to homogeneous coordinates if not already
    if X.shape[0] == 3:
        X_homogeneous = np.vstack((X, np.ones((1, X.shape[1]))))
    else:
        X_homogeneous = X

    # Project points
    projected = M @ X_homogeneous
    # Convert back from homogeneous coordinates
    return projected[:2, :] / projected[2, :]

M1, M1_smallest_singular_value = estimate_camera_DLT(Xmodel, x_image1)
M2, M2_smallest_singular_value = estimate_camera_DLT(Xmodel, x_image2)

projected_points_image1 = project_points(M1, Xmodel)
projected_points_image2 = project_points(M2, Xmodel)

K1, R1 = rq(M1[:, :3])
K2, R2 = rq(M2[:, :3])

K1 /= K1[2, 2]
K2 /= K2[2, 2]

print("Calibration matrix K1:\n", K1)
print("Calibration matrix K2:\n", K2)

print("Camera matrix M:\n", M1)
print("Smallest singular value:", M1_smallest_singular_value)

# Normalize points for the first image
x_image1_normalized, stdx1, stdy1 = normalize_points(x_image1)
x_image2_normalized, stdx2, stdy2 = normalize_points(x_image2)

print("Normalization std for image 1:", stdx1, stdy1)
print("Normalization std for image 2:", stdx2, stdy2)

# Plot the normalized points
plt.scatter(x_image1_normalized[0, :], x_image1_normalized[1, :])
plt.title("Normalized Points for Image 1")
plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.axhline(0, color='gray', lw=0.5)
plt.axvline(0, color='gray', lw=0.5)
plt.grid(True)
plt.show()

# 3D Plot for Camera Centers and Principal Axes

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

camera_center1 = -np.linalg.inv(M1[:, :3]) @ M1[:, 3]
camera_center2 = -np.linalg.inv(M2[:, :3]) @ M2[:, 3]

# extract the principal axis
C1, a1 = camera_center_axis(M1)
C2, a2 = camera_center_axis(M2)

end_point1 = C1 + a1 * 5
end_point2 = C2 + a2 * 5

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the model points
ax.scatter(Xmodel[0, :], Xmodel[1, :], Xmodel[2, :], c='green', marker='o', label='Model Points')

# Plot the actual camera centers
ax.plot((a1[0], end_point1[0]), (a1[1], end_point1[1]), (a1[2], end_point1[2]), "r-")
ax.plot((a2[0], end_point2[0]), (a2[1], end_point2[1]), (a2[2], end_point2[2]), "r-")

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Visualization of Camera Centers and Model Points')
ax.legend()

plt.show()

# -----------------------------------------------   
# Computer exercise 3 & 4

img1 = cv2.imread('ass2/data/CUBE1.jpg')
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

img2 = cv2.imread('ass2/data/CUBE2.jpg')
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()

keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

matches = bf.match(descriptors1, descriptors2)

matches = sorted(matches, key = lambda x:x.distance)

matched_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:50], None, flags=2)

#set size of window smaller
cv2.namedWindow('Matched Features', cv2.WINDOW_NORMAL)
cv2.imshow('Matched Features', matched_img)
cv2.imwrite('matched_features.jpg', matched_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

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

def project_points(points_3D, M):
    """
    Project 3D points back to 2D using the camera matrix.

    :param points_3D: Array of 3D points in homogeneous coordinates (x, y, z, w).
    :param M: Camera matrix.
    :return: Projected 2D points.
    """
    # Exclude the w component (last column) from the 3D points
    points_3D = points_3D[:, :3]

    # Convert to homogeneous coordinates by adding a row of ones
    homogeneous_points = np.vstack((points_3D.T, np.ones((1, points_3D.shape[0]))))

    # Project the points
    points_2D = M.dot(homogeneous_points)

    # Normalize to convert from homogeneous coordinates
    points_2D /= points_2D[2]
    return points_2D[:2].T

keypoints1_homogeneous = np.array([kp.pt + (1,) for kp in keypoints1])
keypoints2_homogeneous = np.array([kp.pt + (1,) for kp in keypoints2])

# Triangulate each pair of points
triangulated_points = []
for i in range(len(matches)):
    idx1 = matches[i].queryIdx
    idx2 = matches[i].trainIdx
    point3D = triangulate_3D_point_DLT(keypoints1_homogeneous[idx1], keypoints2_homogeneous[idx2], M1, M2)
    triangulated_points.append(point3D)

triangulated_points = np.array(triangulated_points)

# Assuming the project_points function is correctly implemented as before

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

# Plotting for the second image with filtered points
plt.figure(figsize=(10, 8))
plt.imshow(img2, cmap='gray')
plt.scatter([kp.pt[0] for kp in keypoints2], [kp.pt[1] for kp in keypoints2], c='r', s=5)
plt.scatter([p[0] for p in filtered_projected_points2], [p[1] for p in filtered_projected_points2], c='b', s=5)
plt.title('Image 2: Original SIFT Points (Red) and Projected Points (Blue)')
plt.show()

K1_inv = np.linalg.inv(K1)
K2_inv = np.linalg.inv(K2)

keypoints1_normalized = np.array([K1_inv @ np.array(kp.pt + (1,)).T for kp in keypoints1])
keypoints2_normalized = np.array([K2_inv @ np.array(kp.pt + (1,)).T for kp in keypoints2])

M1_normalized = K1_inv @ M1
M2_normalized = K2_inv @ M2

triangulated_points = []
for i in range(len(matches)):
    idx1 = matches[i].queryIdx
    idx2 = matches[i].trainIdx
    point3D = triangulate_3D_point_DLT(keypoints1_normalized[idx1][:2], keypoints2_normalized[idx2][:2], M1_normalized, M2_normalized)
    triangulated_points.append(point3D)

triangulated_points = np.array(triangulated_points)

projected_points1 = project_points(triangulated_points, M1)
projected_points2 = project_points(triangulated_points, M2)

# Plotting for the first image with filtered points
plt.figure(figsize=(10, 8))
plt.imshow(img1, cmap='gray')
plt.scatter([kp.pt[0] for kp in keypoints1], [kp.pt[1] for kp in keypoints1], c='r', s=5)
plt.scatter([p[0] for p in filtered_projected_points1], [p[1] for p in filtered_projected_points1], c='b', s=5)
plt.title('Image 1: Original SIFT Points (Red) and Normalized Projected Points (Blue)')
plt.show()

# Plotting for the second image with filtered points
plt.figure(figsize=(10, 8))
plt.imshow(img2, cmap='gray')
plt.scatter([kp.pt[0] for kp in keypoints2], [kp.pt[1] for kp in keypoints2], c='r', s=5)
plt.scatter([p[0] for p in filtered_projected_points2], [p[1] for p in filtered_projected_points2], c='b', s=5)
plt.title('Image 2: Original SIFT Points (Red) and Normalized Projected Points (Blue)')
plt.show()

def compute_pixel_error(projected_points, matched_keypoints, K):
    return np.linalg.norm(projected_points - matched_keypoints, axis=1)

# Create arrays of matched keypoints
matched_keypoints1 = np.array([keypoints1[match.queryIdx].pt for match in matches])
matched_keypoints2 = np.array([keypoints2[match.trainIdx].pt for match in matches])

# Now compute the errors
errors1 = compute_pixel_error(projected_points1, matched_keypoints1, K1)
errors2 = compute_pixel_error(projected_points2, matched_keypoints2, K2)

valid_indices = (errors1 < 3) & (errors2 < 3)
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

# Plot the cube model (assuming Xmodel is in a suitable format)
ax.scatter(Xmodel[0, :], Xmodel[1, :], Xmodel[2, :], c='yellow', marker='s', label='Cube Model', s=20)

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Visualization with Cameras and Cube Model')
ax.legend()

plt.show()

print("Number of filtered 3D points:", len(filtered_3D_points))
if len(filtered_3D_points) > 0:
    print("Sample coordinates:", filtered_3D_points[0])