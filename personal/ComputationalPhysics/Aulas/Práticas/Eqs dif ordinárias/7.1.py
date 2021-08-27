import numpy as np
import matplotlib.pyplot as plt

rc_list = [0.01, 0.1, 1]
t_min = 0; t_max = 10


def f(x, t):
    return 1 - x if int(2 * t) % 2 == 0 else -1 - x


def runge_kutta_4(n, rc):
    h = (t_max - t_min) / n
    t_values = np.linspace(t_min, t_max, n + 1)
    v = 0
    v_values = np.empty(n + 1)
    for i, t in enumerate(t_values):
        v_values[i] = v
        k1 = h * (1 / rc) * f(v, t)
        k2 = h * (1 / rc) * f(v + k1 / 2, t + h / 2)
        k3 = h * (1 / rc) * f(v + k2 / 2, t + h / 2)
        k4 = h * (1 / rc) * f(v + k3, t + h)
        v += (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return t_values, v_values


for rc in rc_list:
    t, v = runge_kutta_4(5000, rc)
    plt.plot(t, v)
    plt.show()
