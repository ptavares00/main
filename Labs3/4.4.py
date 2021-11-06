import matplotlib.pyplot as plt

from Labs3.imports import *

# ----------------------------------------------------------------------------------------------------------------------
# Radiação decaimento 238U

df = pd.read_excel('C:/Users/user/OneDrive - Universidade do Porto/Desktop/4.4.1.xlsx')
mean = df['C'].mean()
std = df['C'].std() / np.sqrt(5)


plt.figure(figsize=(18, 10))
sns.scatterplot(x = np.arange(df['C'].size) + 1, y = df['C'], label='dados experimentais')
plt.hlines(mean, 1, 5, color='orange', label='média')
plt.hlines([2*std + mean, -2*std + mean], 1, 5, color='r', label='limite incerteza média', linestyle='dashed')
plt.grid()
plt.legend(fontsize=15)
plt.xlabel('Medida', fontsize=20)
plt.ylabel(r'C ($s^{-1}$)', fontsize=20)
plt.tick_params(labelsize=15)
plt.xticks([1, 2, 3, 4, 5])
plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# Decaimento dt = 20 s -> Método 2

plt.figure(figsize=(18, 10))
df_20 = pd.read_excel('C:/Users/user/OneDrive - Universidade do Porto/Desktop/4.4.2.xlsx', sheet_name='20')
x = df_20['t'].to_numpy()
y = df_20['N'].to_numpy()
plt.scatter(x, y, s=10, label='dados experimentais')
plt.grid()
plt.tick_params(labelsize=15)
plt.legend(fontsize=15)
plt.xlabel('t (s)', fontsize=20)
plt.ylabel('N', fontsize=20)
plt.show()


results = linest(x[2:15], np.log(y[2:15]))
m = results['m']
b = results['b']
sy_2 = results['2sy']
sm = results['sm']

plt.figure(figsize=(18, 10))
plt.scatter(x=x[2:15], y=np.log(y[2:15]), label='dados experimentais', s=10)
plt.errorbar(x[[2, 14]], x[[2,14]] * m + b, sy_2, color='k', fmt='none', capsize=5)
plt.plot(x[2:15], x[2:15] * m + b, color='orange', label='reta de ajuste')
plt.grid()
plt.legend(fontsize=15)
plt.xlabel('t (s)', fontsize=20)
plt.ylabel('ln(N)', fontsize=20)
plt.tick_params(labelsize=15)
plt.show()

plt.figure(figsize=(18, 10))
sns.residplot(x=x[2:15], y=np.log(y[2:15]), label='dados experimentais')
plt.hlines([sy_2, -sy_2], xmin=50, xmax=300, linestyle='dashed', color='r', label='limites')
plt.grid()
plt.xlabel('t (s)', fontsize=20)
plt.ylabel('Resíduos', fontsize=20)
plt.legend(fontsize=15)
plt.tick_params(labelsize=15)
plt.show()

T = -1 / m * np.log(2)
print(f'T_1/2 = {int(np.round(T))}')
print(f'incerteza = {int(np.round(np.abs(T) * sm / np.abs(m)))}')
print(np.exp(b))


# ----------------------------------------------------------------------------------------------------------------------
# Decaimento dt = 20 s -> Método 2

df_20 = pd.read_excel('C:/Users/user/OneDrive - Universidade do Porto/Desktop/4.4.2.xlsx', sheet_name='20')
t = df_20['t'].to_numpy()
t = t[2:] - t[1]
N_sum = df_20['N'].iloc[2:].cumsum().to_numpy()


def f(t, tau, N0):
    return N0 * (1 - np.exp(-t / tau)) + 0.96 * t


params, cov = curve_fit(f, t, N_sum, [200, 1000])
stderr = cov[np.diag_indices(np.shape(cov)[0])]**0.5

plt.figure(figsize=(18, 10))
plt.scatter(t, N_sum, label='Dados Experimentais')
plt.plot(t, f(t, params[0], params[1]), color='orange', label='Curva de Ajuste')
plt.legend(fontsize=15)
plt.grid()
plt.xlabel('t (s)', fontsize=20)
plt.ylabel('Soma cumulativa de N', fontsize=20)
plt.tick_params(labelsize=15)
plt.show()

plt.figure(figsize=(18, 10))
sns.scatterplot(x=t, y=(N_sum - f(t, params[0], params[1])) / N_sum * 100, s=20)
plt.grid()
plt.xlabel('t (s)', fontsize=20)
plt.ylabel('Resíduos %', fontsize=20)
plt.tick_params(labelsize=15)
plt.show()

print(f'b = {params[1]}')
print(stderr[1])
T = params[0] * np.log(2)
print(f'T_1/2 = {int(np.round(T))}')
print(f'incerteza = {round_number(np.abs(T) * stderr[0] / np.abs(params[0]), 1)}')
