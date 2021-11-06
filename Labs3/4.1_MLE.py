import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as st
from scipy.optimize import minimize


# Load Data
df = pd.read_excel('C:/Users/user/OneDrive - Universidade do Porto/Desktop/Laboratório de Física III/5/4.1.xlsx')
df['delta t (s)'] = df['tempo (s)'].diff()
df.fillna(15.05, inplace=True)
df['A (s^-1)'] = df['N'] / df['delta t (s)']
data = df['N'].to_numpy().flatten()

# Calculate mean and standard deviation
mean = np.mean(data)
print(mean)
std = np.std(data, ddof=1)
print(f'Desvio padrão da média = {np.round(std / data.size**0.5 / mean * 100)}%')

# Find best mu and Poisson Tests
mu_0 = mean
bins_values, bins, patches = plt.hist(data, label="experimental_data", density=True, bins=10, range=[0.5, 10.5])
p_experimental = bins_values


def p_theoretical(mu):
    x = np.arange(data.min(), data.max() + 1)
    return st.poisson.pmf(x, mu)


def l(mu):
    f = st.poisson.pmf(data, mu)
    n = f.size
    return -n * np.mean(np.log(f))


mu_optimal = minimize(l, mu_0)['x'][0]
plt.plot(np.arange(data.min(), data.max() + 1), p_theoretical(mu_optimal),
         marker='o', linestyle='', color='y')
plt.vlines(np.arange(data.min(), data.max() + 1), ymin=0, ymax=p_theoretical(mu_optimal), colors='k', linestyles='-',
           lw=5, label=f"Poisson({np.round(mu_optimal, 3)})", color='y')
plt.xticks(bins[:-1] + 0.5)
lines = plt.vlines(np.linspace(0.5, 10.5, 11), 0, bins_values.max(), color='w', linewidth=0.5)
plt.legend(fontsize=25)
plt.xlabel('N', fontsize=25)
plt.tick_params(labelsize=15)
plt.show()

stat, p = st.kstest(data, 'poisson', args=[mu_optimal])
if p > 0.05:
    print(f'D’Kolmogorov-Smirnov Test: statistic={np.round(stat, 2)}; p={np.round(p, 2)} > 0.05 -> Amostra segue a distribuição'
          f' de poisson (falhou a rejeitar H0)')
else:
    print(f'D’Kolmogorov-Smirnov Test: statistic={np.round(stat, 2)}; p={np.round(p, 2)} < 0.05 -> Amostra não segue a distribuição'
          f' de poisson (falhou a rejeitar H0)')
