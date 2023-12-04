import numpy as np
import matplotlib.pyplot as plt

def psphere(x):
    """
    Normalization of projective points.

    Parameters:
    x : ndarray
        A matrix in which each column is a point in homogeneous coordinates.

    Returns:
    y : ndarray
        Result after normalization.
    alpha : ndarray
        Depth, which is the norm of each point before normalization.
    """
    # Calculate the norm (depth) of each column (point)
    alpha = np.sqrt(np.sum(x**2, axis=0))
    
    # Normalize each column by its norm
    y = x / alpha
    
    return y, alpha


def pflat(x):
    """
    pflat divides each column of an mxn matrix by its last element and outputs the obtained normalized matrix.
    """
    return x[:-1,:] / x[-1,:]

def rital(linjer, st='-'):
    """
    rital takes as input a 3xn matrix "linjer" where each column represents the hom. coordinates of a 2D line.
    It then plots those lines. Use "plt.hold(True)" before rital to see all the lines.
    The optional second argument "st" controls the line style of the plot.
    """
    if linjer.size == 0:
        return  # If linjer is empty, do nothing

    nn = linjer.shape[1]
    rikt = psphere(np.array([linjer[1, :], -linjer[0, :], np.zeros(nn)]))
    punkter = pflat(np.cross(rikt.T, linjer.T).T)

    for i in range(nn):
        plt.plot([punkter[0, i] - 2000 * rikt[0, i], punkter[0, i] + 2000 * rikt[0, i]],
                 [punkter[1, i] - 2000 * rikt[1, i], punkter[1, i] + 2000 * rikt[1, i]], st)
