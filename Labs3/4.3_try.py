import matplotlib.pyplot as plt
import numpy as np

from Labs3.imports import *


def Omega(r):
    R = 0.0045
    return 4 * np.arctan(R / r) * R / (r**2 + R**2) ** 0.5


df = pd.read_excel('C:/Users/user/OneDrive - Universidade do Porto/Desktop/4.3.xlsx')
df['C'] = df[['N1', 'N2', 'N3']].mean(skipna=True, axis=1) / 10
df['C/O'] = df['C'] / Omega(df['d (m)'])

x_ajuste = df['d (m)'].to_numpy()
y_real = df['C/O'].to_numpy()

sns.scatterplot(x=x_ajuste, y=y_real)
plt.show()

# ----------------------------------------------------------------------------------------------------------------------

fig1, ax1 = plt.subplots(1, 2)

res = linest(x_ajuste[:4], y_real[:4])
sns.scatterplot(x=x_ajuste[:], y=y_real[:], ax=ax1[1], label='dados experimentais')
sns.scatterplot(x=x_ajuste[:4], y=y_real[:4], ax=ax1[1], label='pontos ajuste', color='orange')
x = np.linspace(x_ajuste[0], x_ajuste[-1], 1000)
sns.lineplot(x=x,
             y=x * res.slope + res.intercept,
             linestyle="dashed", color='orange', ax=ax1[1], label='reta ajuste')

alpha_range = - res.intercept / res.slope
alpha_range_uncertainty = alpha_range * np.sqrt((res.stderr / res.slope)**2 + (res.intercept_stderr / res.intercept)**2)
print(f'coeficiente de correlação: {res.rvalue**2}')
print(f'm: {res.slope}')
print(f'sm: {res.stderr}')
print(f'b: {res.intercept}')
print(f'sb: {res.intercept_stderr}')
print(f'alcance: {round_number(alpha_range, 2)}')
print(f'incerteza: {round_number(alpha_range_uncertainty, 1)}')
print(f'incerteza = {int(round_number(alpha_range_uncertainty / alpha_range * 100, 2))} %')

sns.scatterplot(x=[alpha_range], y=[0], ax=ax1[1], label='alcance', color='r', s=50)
# ax1[1].set_ylim(-20, 100)
ax1[1].set_xlabel('d (m)', fontsize=20)
ax1[1].set_ylabel(r'C ($s^{-1}$)', fontsize=20)
ax1[1].grid()
ax1[1].legend(fontsize=15)
ax1[1].tick_params(labelsize=15)

sns.scatterplot(x=x_ajuste[:], y=y_real[:], ax=ax1[0], label='dados experimentais')
ax1[0].set_xlabel('d (m)', fontsize=20)
ax1[0].set_ylabel(r'C ($s^{-1}$)', fontsize=20)
ax1[0].grid()
ax1[0].legend(fontsize=15)
ax1[0].tick_params(labelsize=15)

plt.show()

