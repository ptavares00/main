import numpy as np
import matplotlib.pyplot as plt


def v(x):
    v0 = 50 * 1.602e-19; a = 1e-11
    return v0 * x**4 / a**4


def f(r, x, E):
    m = 9.11e-31; h = 1.054571e-34
    psi, phi = r
    f_psi = phi
    f_phi = (2 * m / h**2) * (v(x) - E) * psi
    return np.array([f_psi, f_phi], float)


def solution(E):
    E = E * 1.602e-19 * 1e6
    a = 1e-11
    x_min = -10 * a; x_max = 10 * a
    n = 50000
    h = (x_max - x_min) / n
    x_values = np.linspace(x_min, x_max, n+1)
    r = np.array([0, 1], float)
    r_values = np.empty([n+1, 2], float)
    for i, x in enumerate(x_values):
        r_values[i] = r
        k1 = h * f(r, x, E)
        k2 = h * f(r + k1/2, x + h/2, E)
        k3 = h * f(r + k2/2, x + h/2, E)
        k4 = h * f(r + k3, x + h, E)
        r += (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return x_values, r_values[:, 0]


E0 = 1.00321074; E1st = 3.50835188; E2nd = 5.93835543

# Ground state
x_list, psi = solution(E0)
plt.plot(x_list, psi)
plt.show()

# 1st excited state
x_list, psi = solution(E1st)
plt.plot(x_list, psi)
plt.show()

# 2nd excited state
x_list, psi = solution(E2nd)
plt.plot(x_list, psi)
plt.show()

