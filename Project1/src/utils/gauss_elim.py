"""
Implementations of Gaussian elimination.
"""

import numpy as np

def gaussian_elimination(A, b):
    """General Gaussian elimination. Solve Av = b, for `v`.
    `A` is a square matrix with dimensions (n,n) and `b` has dim (n,)

    Parameters
    ----------
    A : np.ndarray
        (n,n) matrix

    b : np.ndarray
        (n,) RHS of linear equation set.

    Returns
    -------
    np.ndarray
        The solution `v` for Av = b.
    """
    # Join A and b
    ab = np.c_[A,b]
    # Gaussian Elimination
    for i in range(n-1):
        if ab[i,i] == 0:
            raise ZeroDivisionError('Zero value in matrix..')

        for j in range(i+1, n):
            ratio = ab[j,i] / ab[i,i]

            for k in range(i, n+1):
                ab[j,k] = ab[j,k] - ratio * ab[i,k]

    # Backward Substitution
    X = np.zeros((n,1))
    X[n-1,0] = ab[n-1,n] / ab[n-1,n-1]

    for i in range(n-2,-1,-1):
        knowns = ab[i, n]
        for j in range(i+1, n):
            knowns -= ab[i,j] * X[j,0]
        X[i,0] = knowns / ab[i,i]
    return X

def gaussian_elimination_special_case(b):
    """Gaussian elimination specialised for solving
    Av = b, where `A` is a tridiagonal matrix where
    all elements on the diagonal are 2, and the
    elements on the adjacent "diagonal" are -1.

    Parameters
    ----------
    b : np.ndarray
        The response

    Returns
    -------
    np.ndarray
        The solution `v` for Av = b.
    """
    n = len(b)
    # init new (prime) arrays
    beta_prime = np.empty(n)
    beta_prime[0] = 2

    b_prime = np.empty(n)
    b_prime[0] = b[0]

    v = np.empty(n)
    i_array = np.arange(n)
    beta_prime = (i_array+2) / (i_array+1)

    for i in range(1,n):
        b_prime[i] = b[i] + (b_prime[i-1] / beta_prime[i-1])

    v[-1] = b_prime[-1] / beta_prime[-1]

    for i in range(n-2, -1, -1):
        v[i] = (b_prime[i] + v[i+1])/ beta_prime[i]

    return v
