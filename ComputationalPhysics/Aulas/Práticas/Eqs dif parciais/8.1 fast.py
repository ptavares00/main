import numpy as np
import matplotlib.pyplot as plt

n = 200


def make_rho(m):
    i_min = int(0.6 * n+1); i_max = int(0.8 * n+1)
    j_min = int(0.2 * n+1); j_max = int(0.4 * n+1)
    m[i_min: i_max, i_min: i_max] = -1
    m[j_min: j_max, j_min: j_max] = 1


def jacobi(n):
    a = 1 / n  # x_max = 100 cm / n = 100 points
    phi1 = np.zeros([n+1, n+1], float)
    phi2 = phi1.copy()

    rho = np.zeros([n+1, n+1])
    make_rho(rho)

    error = 1
    precision = 1e-8

    while error > precision:
        aux1 = (phi1[:-2, 1:-1] + phi1[2:, 1:-1] + phi1[1:-1, :-2] + phi1[1:-1, 2:]) / 4
        aux2 = a**2 * rho[1:-1, 1:-1] / 4
        phi2[1:-1, 1:-1] = aux1 + aux2
        error = np.max(abs(phi2 - phi1))
        phi1 = phi2.copy()
    return phi2


solution = jacobi(n)
plt.imshow(solution, origin="lower")
plt.colorbar()
plt.show()
