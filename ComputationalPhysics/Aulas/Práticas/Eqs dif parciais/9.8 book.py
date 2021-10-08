import numpy as np
import cmath as cm
import scipy.linalg as sp
import matplotlib.pyplot as plt

L = 1e-8; M = 9.109e-31; hbar = 1.05e-34
N = 4000; a = L / N; h = 1e-18


def initial_psi(x):
    sigma = 1e-10; k = 5e10; x0 = L / 2
    return cm.exp(- (x - x0) ** 2 / (2 * sigma ** 2)) * cm.exp(1j * k * x)
initial_psi = np.vectorize(initial_psi)


# Matriz A
a1 = 1 + (h * hbar * 1j) / (2 * M * a ** 2)
a2 = - (h * hbar * 1j) / (4 * M * a ** 2)
A = np.empty((3, N+1), complex)
A[[0, 2], :] = a2; A[1] = a1
A[0, 0] = A[-1, -1] = 0

# Matriz B
b1 = 1 - (h * hbar * 1j) / (2 * M * a ** 2)
b2 = - a2
B = np.identity(N + 1) * b1 + np.eye(N+1, k=1) * b2 + np.eye(N+1, k=-1) * b2


def solution():
    x_values = np.linspace(0, L, N + 1)
    t_max = 1e-15
    t_values = np.arange(0, t_max, h)
    psi = initial_psi(x_values)
    psi_values = np.empty((len(t_values), N + 1), complex)
    for i, t in enumerate(t_values):
        psi_values[i] = psi
        psi = sp.solve_banded((1, 1), A, B @ psi)
    return x_values, psi_values


x_list, psi_list = solution()
psi_last = psi_list[-1]
plt.plot(x_list, abs(psi_list[int(len(psi_list) / 2)]))
plt.plot(x_list, abs(psi_last))
plt.show()
