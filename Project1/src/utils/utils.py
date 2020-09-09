"""Utility functions
"""
import numpy as np


def tridiag(a, b, c, k1=-1, k2=0, k3=1):
    """Create tridiagonal matrix from vectors"""
    return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)

def error(v,u):
    """Log10 relative errors between elements of `u` and `v`"""
    return np.log10(np.abs((v-u)/u))
