import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import numba as nb

n = 200
w = 0.9


@jit(nopython=True)
def gauss_seidel(n, w):
    phi = np.zeros((n + 1, n + 1), dtype=nb.float64)
    k1 = int(n * 0.2); k2 = int(n * 0.8)
    phi[k1:k2, k1] = 1
    phi[k1:k2, k2] = -1
    error = 1
    precision = 1e-6
    while error > precision:
        error = 0
        for i in range(n+1):
            for j in range(n+1):
                if i == 0 or i == n or j == 0 or j == n:
                    continue
                elif (j == k1 or j == k2) and (k1 <= i < k2):
                    continue
                else:
                    old = phi[i, j]
                    phi[i, j] = (1 + w) * (phi[i-1, j] + phi[i+1, j] + phi[i, j-1] + phi[i, j+1]) / 4 - w * phi[i, j]
                    new_error = abs(phi[i, j] - old)
                    error = max(new_error, error)
    return phi


solution = gauss_seidel(n, w)
plt.imshow(solution, origin="lower")
plt.colorbar()
plt.show()
