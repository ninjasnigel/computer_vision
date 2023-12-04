import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import color

def pflat(x):
    return x[:-1,:] / x[-1,:]

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


# Load data from data/compEx3im1.jpg
im1 = plt.imread('data/compEx3im1.jpg')
# Load data from data/compEx3im2.jpg
im2 = plt.imread('data/compEx3im2.jpg')

#plot images side by side
fig, ax = plt.subplots(1, 2)
ax[0].imshow(im1, cmap='gray')
ax[1].imshow(im2, cmap='gray')
plt.show()

# Load data from data/compEx3.mat
mat = scipy.io.loadmat('data/compEx3.mat')

# Calculate the camera centers and principal axes of the two cameras
P1, P2, U = mat['P1'], mat['P2'], mat['U']
C1, a1 = camera_center_axis(P1)
C2, a2 = camera_center_axis(P2)

# print Camera centers in cartesian coordinates
print("Camera center 1: ", C1)
print("Camera center 2: ", C2)

# Print prinicpal axes normalized to one
print("Principal axis 1: ", a1)
print("Principal axis 2: ", a2)

# Convert U to non-homogeneous coordinates
U_nh = pflat(U)

# Set up a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D points
ax.scatter(U_nh[0,:], U_nh[1,:], U_nh[2,:], c='b', marker='.')

# Plot the camera centers
plot_camera(P1, s=5, ax=ax)
plot_camera(P2, s=5, ax=ax)

ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.set_title('3D points and Camera Centers')

plt.show()