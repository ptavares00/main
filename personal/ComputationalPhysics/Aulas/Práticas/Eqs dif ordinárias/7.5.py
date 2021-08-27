import numpy as np
import matplotlib.pyplot as plt

g = 9.8; l = 0.1; c = 2
# greek = 5  # pedido
greek = 10  # ressonância
t_min = 0; t_max = 100


def f(r, t):
    theta, omega = r
    f1 = omega
    f2 = - g / l * np.sin(theta) + c * np.cos(theta) * np.sin(greek * t)
    return np.array([f1, f2], float)


def runge_kutta_4(n):
    h = (t_max - t_min) / n
    t_values = np.linspace(t_min, t_max, n+1)
    r = np.array([0, 0], float)
    r_values = np.empty([n+1, 2], float)
    for i, t in enumerate(t_values):
        r_values[i] = r
        k1 = h * f(r, t)
        k2 = h * f(r + k1/2, t + h/2)
        k3 = h * f(r + k2/2, t + h/2)
        k4 = h * f(r + k3, t + h)
        r += (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return t_values, r_values


t_list, r_list = runge_kutta_4(20000)
theta = r_list[:, 0]
plt.plot(t_list, theta)
plt.show()