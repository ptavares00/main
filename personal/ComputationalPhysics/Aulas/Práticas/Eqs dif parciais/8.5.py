import numpy as np
import matplotlib.pyplot as plt

l = 1; d = 0.1; c = 1; sigma = 0.3
n = 100; a = l / n
v = 500
h = 1e-6

x_values = np.linspace(0, l, n + 1)
t_max = 0.1
t_values = np.arange(0, t_max, h)

r_values = np.empty((len(t_values), n + 1, 2), float)
r = np.zeros((n + 1, 2), float)


def initial_psi(x):
    aux1 = c * x * (l - x) / l ** 2
    aux2 = np.exp(- (x - d) ** 2 / (2 * sigma ** 2))
    return aux1 * aux2


r[:, 1] = initial_psi(x_values)

for i, t in enumerate(t_values):
    r_values[i] = r
    phi = r[:, 0].copy(); psi = r[:, 1].copy()
    r[1:-1, 0] += h * psi[1:-1]
    r[1:-1, 1] += h * (v / a) ** 2 * (phi[:-2] + phi[2:] - 2 * phi[1:-1])

    if i % 500 == 0:
        plt.plot(r[:, 0])
        plt.show()
