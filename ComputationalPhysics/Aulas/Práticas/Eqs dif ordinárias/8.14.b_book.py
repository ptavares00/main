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
    return r_values[-1, 0]
#     return x_values, r_values[:, 0]
# x_list, psi = solution(138.02424631)
# plt.plot(x_list, psi)
# plt.show()


# programa concebido para receber E em Mev. Dentro transforma em Joules.
def find_energy(E1, E2):
    precision = 1e-8
    error = abs(E1 - E2)
    psi1 = solution(E1)
    while error > precision:
        E_average = (E1 + E2) / 2
        psi_average = solution(E_average)
        if psi_average * psi1 > 0:
            E1 = E_average
            psi1 = solution(E1)
        else:
            E2 = E_average
        error = abs(E1 - E2)
    return (E1 + E2) / 2


print("")

E1 = 1; E2 = 2
E0 = find_energy(E1, E2)
print("ground state energy: {} MeV".format(round(E0, 8)))

E1 = 2; E2 = 4
E1st = find_energy(E1, E2)
print("1st excited state energy: {} MeV".format(round(E1st, 8)))

E1 = 4; E2 = 6
E2nd = find_energy(E1, E2)
print("2nd excited state energy: {} MeV".format(round(E2nd, 8)))
