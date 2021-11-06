from Labs3.imports import *


def f(x, a, b):
    return a / x**2 + b


df = pd.read_excel('C:/Users/user/OneDrive - Universidade do Porto/Desktop/4.3.xlsx')
df['C'] = df[['N1', 'N2', 'N3']].mean(skipna=True, axis=1) / 10 - 0.36
df['d^-2'] = df['d (m)']**-2

x_ajuste = df['d (m)'].to_numpy()
y_real = df['C'].to_numpy()
# params, _ = curve_fit(f, x_ajuste, y_real)
# y_ajuste = f(x_ajuste, params[0], params[1])
# ni = x_ajuste.size - 14
# nf = x_ajuste.size
# sigma_y = np.std(y_real[ni:nf], ddof=1) / y_real.mean()
# res = fit_table(x_ajuste, y_real)
# print(f'coeficiente de determinação: {round_number(res.rvalue**2, 4)}')
# print(f'declive: {round_number(res.slope, 4)}')
# activity = 4 * np.pi * res.slope / 0.635e-4
# print(f'Atividade total: {round_number(activity, 4)}')
# uncertainty = activity * ((res.stderr / res.slope)**2 + (0.001 / 0.635)**2)**0.5
#
#
# fig, ax = plt.subplots(1)
# fig.set_size_inches(20, 10)
# fig.suptitle(rf'Atividade total = {int(round_number(activity, 4))} $s^{{-1}}$ $\pm$ {round_number(uncertainty / activity, 1)} %',
#              fontsize='xx-large')
#
# sns.scatterplot(x=x_ajuste[5:nf], y=y_real[5:nf], ax=ax, label='dados experimentais')
# sns.scatterplot(x=x_ajuste[ni:nf], y=y_real[ni:nf], ax=ax, label='dados usados no ajuste', color='orange')
# sns.lineplot(x=x_ajuste[ni:nf],
#              y=x_ajuste[ni:nf] * res.slope + res.intercept,
#              linestyle="dashed", color='orange', ax=ax, label='reta ajuste')
# ax.tick_params(labelsize=15)
# ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
# # ax[0].set_xscale('log')
# # ax[0].set_yscale('log')
# ax.set_xlabel(r'$d^{-2}$ ($m^{-2}$)', fontsize=20)
# ax.set_ylabel(r'C ($s^{-1}$)', fontsize=20)
# ax.grid()
# ax.legend(fontsize=15)
# plt.show()
#
# fig, ax = plt.subplots(1, 2)
# fig.set_size_inches(20, 10)
# fig.suptitle(rf'Atividade total = {int(round_number(activity, 4))} $s^{{-1}}$ $\pm$ {round_number(uncertainty / activity, 1)} %',
#              fontsize='xx-large')
#
# sns.regplot(x=x_ajuste[ni:nf], y=y_real[ni:nf], ax=ax[0], label='dados experimentais', x_ci=None)
# ax[0].tick_params(labelsize=15)
# ax[0].ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
# ax[0].set_xlabel(r'$d^{-2}$ ($m^{-2}$)', fontsize=20)
# ax[0].set_ylabel(r'C ($s^{-1}$)', fontsize=20)
# ax[0].grid()
# ax[0].set_xlim(10, 1.5e3)
# ax[0].legend(fontsize=15)
#
# sns.residplot(x=x_ajuste[ni:nf], y=y_real[ni:nf], ax=ax[1])
# ax[1].hlines([-2 * sigma_y, 2 * sigma_y], xmin=x_ajuste[ni], xmax=x_ajuste[nf-1], linestyles='dashed',
#              label='limites', color='r')
# ax[1].tick_params(labelsize=15)
# ax[1].ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
# ax[1].set_xlabel(r'$d^{-2}$ ($m^{-2}$)', fontsize=20)
# ax[1].set_ylabel(r'Resíduos %', fontsize=20)
# ax[1].grid()
# ax[1].set_xlim(10, 1.5e3)
# ax[1].legend(fontsize=15)
#
# plt.show()


# Alcance de partículas alpha
fig1, ax1 = plt.subplots(1, 2)

res = fit_table(x_ajuste[:4], y_real[:4])
sns.scatterplot(x=x_ajuste[:15], y=y_real[:15], ax=ax1[1], label='dados experimentais')
sns.scatterplot(x=x_ajuste[:4], y=y_real[:4], ax=ax1[1], label='pontos ajuste', color='orange')
x = np.linspace(x_ajuste[0], x_ajuste[10], 100)
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
ax1[1].set_ylim(-20, 100)
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
