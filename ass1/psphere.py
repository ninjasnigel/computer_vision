import numpy as np

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

# Example usage:
x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # Example input
y, alpha = psphere(x)
print("Normalized points (y):")
print(y)
print("Depths (alpha):")
print(alpha)
