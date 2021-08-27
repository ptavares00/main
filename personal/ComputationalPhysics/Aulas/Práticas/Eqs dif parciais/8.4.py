import numpy as np
import matplotlib.pyplot as plt


def initial_temperature(t):
    return 10 + 12 * np.sin(2 * np.pi * t / 365)


n = 200; a = 20 / n
D = 0.1
h = 0.1; epsilon = h / 1000
t_max = 10 * 365 + epsilon

t1 = 9 * 365 + int(3 / 12 * 365)
t2 = 9 * 365 + int(6 / 12 * 365)
t3 = 9 * 365 + int(9 / 12 * 365)
t4 = 10 * 365

temperature = np.ones(n + 1, float) * 10
temperature[-1] = 11
t_values = np.arange(0, 10 * 365 + epsilon, h)
temperature_values = np.empty((len(t_values), n+1), float)
indexes = []

for i, t in enumerate(t_values):
    temperature[0] = initial_temperature(t)
    temperature_values[i] = temperature

    temperature[1:-1] += h * D / a**2 * (temperature[:-2] + temperature[2:] - 2 * temperature[1:-1])

    if abs(t - t1) < epsilon:
        indexes.append(i)
    elif abs(t - t2) < epsilon:
        indexes.append(i)
    elif abs(t - t3) < epsilon:
        indexes.append(i)
    elif abs(t - t4) < epsilon:
        indexes.append(i)

for i in indexes:
    plt.plot(np.linspace(0, 20, n+1), temperature_values[i], label="index = {}".format(i))
plt.grid()
plt.legend()
plt.show()

