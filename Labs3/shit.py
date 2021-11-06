import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def f(x, a, b):
    return a / x**2 + b


df = pd.read_excel('C:/Users/user/OneDrive - Universidade do Porto/Desktop/4.3.xlsx')
df['C'] = df[['N1', 'N2', 'N3']].mean(skipna=True, axis=1) / 10
x_ajuste = df['d (m)'].to_numpy()[:30]
y_real = df['C'].to_numpy()[:30]


x = np.linspace(0.001, 0.030, 2000)
plt.scatter(x_ajuste, y_real, label='exp')
plt.plot(x, f(x, 1e-5, 0), label=1)
plt.plot(x, f(x, 0.5e-5, 0), label=0.5)
plt.plot(x, f(x, 2e-5, 0), label=2)
plt.plot(x, f(x, 5e-4, 10), label=10)
plt.plot(x, f(x, 0.2e-5, 0), label=0.2)
plt.legend(
)
plt.show()
