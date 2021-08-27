import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import numba as nb
# Não consegui implementar com métodos do numpy. Ficava sempre como o de Jacobi. Assim, usei o numba e consegui tempos
# muito pequenos.

n = 300
w = 0.9


@jit(nopython=True)
def gauss_seidel(n, w):
    phi = np.zeros((n + 1, n + 1), dtype=nb.float64)
    phi[-1, :] = 1
    error = 1
    precision = 1e-6
    while error > precision:
        error = 0
        for i in range(n+1):
            for j in range(n+1):
                if i == 0 or i == n or j == 0 or j == n:
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
