import numpy as np

from FisicaEstatistica.TPC1.generate_matrix import *
import time


def weigner_semicircle(x, n):
    r_2 = 8 * n
    return 2 / (np.pi * r_2) * np.sqrt(r_2 - x**2)


N_list = [4, 32, 64, 128]
fig, axes = plt.subplots(2, 2)
fig.set_size_inches(16, 16)
axes_list = [(0, 0), (0, 1), (1, 0), (1, 1)]
for i, N in enumerate(N_list):
    N_matrix = GUE(N, 3000)
    eig_values = np.array([])
    for matrix in N_matrix:
        eig_values = np.concatenate([eig_values, np.linalg.eigvalsh(matrix)])
    x, y = histogram(eig_values, -40, 40, 200)
    axes[axes_list[i]].plot(x, y, label=f"calculated_{N}")
    x = np.linspace(-40, 40, 1000)
    axes[axes_list[i]].plot(np.linspace(-40, 40, 1000), weigner_semicircle(x, N), label=f"expected_{N}")
    axes[axes_list[i]].legend()
plt.show()


