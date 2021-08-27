import numpy as np
import matplotlib.pyplot as plt

g = 9.8; l = 0.1
t_min = 0; t_max = 10


def f(r, t):  # exemplo de função
    theta, omega = r
    f1 = omega
    f2 = - g / l * np.sin(theta)
    return np.array([f1, f2], float)


def runge_kutta_4(n):
    h = (t_max - t_min) / n  # espaçamento temporal
    t_values = np.linspace(t_min, t_max, n+1)  # lista com os valores do tempo
    r = np.array([179 * np.pi / 180, 0], float)  # condições iniciais
    r_values = np.empty([n+1, 2], float)  # array para armazenar a solução e a derivada para os diferentes t's
    for i, t in enumerate(t_values):
        r_values[i] = r  # guardar o valor calculado

        # calcular o valor seguinte com o método de runge-kuta-4
        k1 = h * f(r, t)
        k2 = h * f(r + k1/2, t + h/2)
        k3 = h * f(r + k2/2, t + h/2)
        k4 = h * f(r + k3, t + h)
        r += (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return t_values, r_values


t_list, r_list = runge_kutta_4(1000)
theta = r_list[:, 0]
plt.plot(t_list, theta)
plt.show()
