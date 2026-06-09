import numpy as np


def tdma(a, b, c, d):
    """
    Solve a tridiagonal linear system using the Thomas algorithm.

    The system has the form:

        a[i] x[i-1] + b[i] x[i] + c[i] x[i+1] = d[i]

    Parameters
    ----------
    a : ndarray
        Lower diagonal.
    b : ndarray
        Main diagonal.
    c : ndarray
        Upper diagonal.
    d : ndarray
        Right-hand side vector.

    Returns
    -------
    x : ndarray
        Solution vector.
    """
    n = len(d)

    cp = np.zeros(n)
    dp = np.zeros(n)
    x = np.zeros(n)

    cp[1] = c[1] / b[1]
    dp[1] = d[1] / b[1]

    for i in range(2, n - 1):
        denominator = b[i] - a[i] * cp[i - 1]

        cp[i] = c[i] / denominator
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denominator

    x[n - 2] = dp[n - 2]

    for i in range(n - 3, 0, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]

    return x