from Labs3.imports import *
from seaborn_qqplot import pplot

# Load Data
df = pd.read_excel('C:/Users/user/OneDrive - Universidade do Porto/Desktop/Laboratório de Física III/5/4.1.xlsx')
df['delta t (s)'] = df['tempo (s)'].diff()
df.fillna(15.05, inplace=True)
df['A (s^-1)'] = df['N'] / df['delta t (s)']
data = df['A (s^-1)'].to_numpy().flatten()

# Calculate mean and standard deviation
mean = np.mean(data)
print(f'média = {mean}')
std = np.std(data, ddof=1)
print(f'Desvio padrão da média = {np.round(std / data.size**0.5 / mean * 100)}%')

# Gaussian Tests
stat, p = st.normaltest(data)
print(f'D’Agostino’s K^2 Test: statistic={np.round(stat, 2)}; p={np.round(p, 2)} > 0.05 -> Amostra parece gaussiana (falhou a rejeitar H0)')
stat, p = st.shapiro(data)
print(f'Shapiro-Wilk Test: statistic={np.round(stat, 2)}; p={np.round(p, 2)} > 0.05 -> Amostra parece gaussiana (falhou a rejeitar H0)')

# Histogram and qqplot of Normal distribution
mu, sigma = st.norm.fit(data)
x = np.linspace(data.min(), data.max(), len(data)*5)
pdf = st.norm.pdf(x, mu, sigma)

fig = plt.figure(figsize=(12, 8))

plt.hist(data, density=True, label="dados experimentais",)
plt.plot(x, pdf, label=f'N({np.round(mu, 2)}, {np.round(sigma, 2)})')
plt.tick_params(labelsize=15)
plt.legend(fontsize=25)
plt.xlabel(r'A ($s^{-1}$)', fontsize=25)
plt.show()

data_df = np.zeros((data.size, 2))
data_df[:, 1] = data
data_df[:, 0] = st.norm.rvs(size=data.size)
df = pd.DataFrame(data_df, columns=["dados teóricos", 'dados experimentais'])

pplot(df, x='dados teóricos', y='dados experimentais', kind='qq', height=8, aspect=1.5,
      display_kws={"identity": False, "fit": True, "reg": True, "ci": 0.05},)
plt.show()
