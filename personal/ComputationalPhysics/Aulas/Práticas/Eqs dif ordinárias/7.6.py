import numpy as np
import matplotlib.pyplot as plt

omega = 1; t_min = 0; t_max = 50


def f1(r, t):
    x, y = r
    f1 = y
    f2 = - omega ** 2 * x
    return np.array([f1, f2], float)


def f3(r, t):
    x, v = r
    f1 = v
    f2 = - omega ** 2 * x ** 3
    return np.array([f1, f2], float)


def van(r, t):
    x, v = r
    f1 = v
    f2 = u * (1 - x ** 2) * v - omega ** 2 * x
    return np.array([f1, f2], float)


def runge_kutta_4(n, f):
    h = (t_max - t_min) / n
    t_values = np.linspace(t_min, t_max, n+1)
    r = np.array([1, 0], float)
    r_values = np.empty([n+1, 2], float)
    for i, t in enumerate(t_values):
        r_values[i] = r
        k1 = h * f(r, t)
        k2 = h * f(r + k1 / 2, t + h / 2)
        k3 = h * f(r + k2 / 2, t + h / 2)
        k4 = h * f(r + k3, t + h)
        r += (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return t_values, r_values


# 1 e 2
t_list, r_list = runge_kutta_4(5000, f1)
x = r_list[:, 0]
plt.plot(t_list, x)
plt.show()


# 3 e 4
t_list, r_list = runge_kutta_4(5000, f3)
x = r_list[:, 0]; v = r_list[:, 1]

plt.figure()
plt.plot(t_list, x)
plt.show()

plt.figure()
plt.plot(x, v)
plt.show()


# 5
omega = 1; u = 4; t_min = 0; t_max = 20
t_list, r_list = runge_kutta_4(5000, van)
x = r_list[:, 0]; v = r_list[:, 1]

plt.figure()
plt.plot(x, v)
plt.show()
