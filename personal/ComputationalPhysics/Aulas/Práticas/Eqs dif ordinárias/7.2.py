import numpy as np
import matplotlib.pyplot as plt

alpha = 1; beta = gamma = 0.5; delta = 2
t_min = 0; t_max = 30


def f(r, t):
    x = r[0]; y = r[1]
    fx = alpha * x - beta * x * y
    fy = gamma * x * y - delta * y
    return np.array([fx, fy], float)


def runge_kutta_4(n):
    h = (t_max - t_min) / n
    t_values = np.linspace(t_min, t_max, n+1)
    r = np.array([2, 2], float)
    r_values = np.empty([n+1, 2], float)
    for i, t in enumerate(t_values):
        r_values[i] = r
        k1 = h * f(r, t)
        k2 = h * f(r + k1/2, t + h/2)
        k3 = h * f(r + k2/2, t + h/2)
        k4 = h * f(r + k3, t + h)
        r += (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return t_values, r_values


t_list, r_list = runge_kutta_4(1000)
rabbits = r_list[:, 0]
foxes = r_list[:, 1]

plt.plot(t_list, rabbits, label="rabbits")
plt.plot(t_list, foxes, label="foxes")
plt.legend()
plt.show()
