import numpy as np
import matplotlib.pyplot as plt

sigma = 10; r = 28; b = 8 / 3
t_min = 0; t_max = 50


def f(u, t):
    x, y, z = u
    fx = sigma * (y - x)
    fy = r * x - y - x * z
    fz = x * y - b * z
    return np.array([fx, fy, fz], float)


def runge_kutta_4(n):
    h = (t_max - t_min) / n
    t_values = np.linspace(t_min, t_max, n+1)
    u = np.array([0, 1, 0], float)
    u_values = np.empty([n+1, 3], float)
    for i, t in enumerate(t_values):
        u_values[i] = u
        k1 = h * f(u, t)
        k2 = h * f(u + k1 * 2, t + h/2)
        k3 = h * f(u + k2 * 2, t + h / 2)
        k4 = h * f(u + k3, t + h)
        u += (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return t_values, u_values


t_list, u_list = runge_kutta_4(500000)
x = u_list[:, 0]; y = u_list[:, 1]; z = u_list[:, 2]

plt.figure()
plt.plot(t_list, y)
plt.show()

plt.figure()
plt.plot(x, z)
plt.show()
