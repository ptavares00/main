from FisicaEstatistica.TPC1.generate_matrix import *
import time


def rho_5(x):
    return 32 * x**2 / np.pi**2 * np.exp(-4 * x**2 / np.pi)


N_list = [4, 32, 64, 128]
for N in N_list:
    N_matrix = GUE(N, 3000)
    eig_values_spaces = np.array([])
    for matrix in N_matrix:
        eig_values = np.linalg.eigvalsh(matrix)
        spaces = eig_values[1:] - eig_values[:-1]
        eig_values_spaces = np.concatenate([eig_values_spaces, spaces / spaces.mean()])
    x, y = histogram(eig_values_spaces, 0, 10, 200)
    plt.plot(x, y, label=f"calculated_{N}")
    x = np.linspace(0, 10, 1000)
    plt.plot(np.linspace(0, 10, 1000), rho_5(x), label=f"expected_{N}")
    plt.legend()
    plt.show()
