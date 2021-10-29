import numpy as np
import scipy
import numba

def householder_elim(q, a, i_min):
    N = len(a[:, 0]) - i_min
    e0_v = np.zeros(N); e0_v[0] = 1
    u_v = np.ndarray.copy(a[i_min:, i_min])
    u_v += np.sqrt(u_v @ u_v) * e0_v
    u_v /= np.sqrt(u_v @ u_v)
    a[i_min:, i_min:] -= 2 * np.outer(u_v, np.dot(u_v, a[i_min:, i_min:]))
    q[:, i_min:] -= 2 * np.outer(q[:, i_min:] @ u_v, u_v)


def householder_qr(a):
    n = len(a[:, 0])
    r = np.ndarray.copy(a)
    q = np.eye(n)
    for i in range(n-1):
        householder_elim(q, r, i)
    return q, r


# Determina os valores próprios
def eigen_values(a, p):
    a1 = np.ndarray.copy(a)
    n = len(a)
    precision_reached = False

    while not precision_reached:
        q, r = householder_qr(a1)
        a1 = r @ q
        precision_reached = not (False in (abs(a1[~np.eye(n, dtype=bool)]) < p))

    return np.diag(a1)
