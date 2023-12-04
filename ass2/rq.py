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

# Example usage
# a = np.array([[...]])  # Replace with actual matrix data
# r, q = rq(a)
