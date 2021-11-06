from Labs3.imports import *

for include in [[100], [200], [100, 200], [500], [100, 200, 500]]:
    # Load Data
    for i, n in enumerate(include):
        df = pd.read_excel('C:/Users/user/OneDrive - Universidade do Porto/Desktop/4.2.2.xlsx', str(n))
        df['delta t (s)'] = df['t (s)'].diff().fillna(df['t (s)'].iloc[0])
        df['A (s^-1)'] = df['N'] / df['delta t (s)']
        data = df['A (s^-1)'].to_numpy().flatten()
        if i == 0:
            df_ = df.copy()
        else:
            df_ = pd.concat([df_, df])
    df = df_.copy()
    data = df['N'].to_numpy().flatten()

    # Calculate mean and standard deviation
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    print(mean)
    print(f'Desvio padrão da média = {np.round(std / data.size**0.5 / mean * 100)}%')

    # Find best mu and Poisson Tests
    mu_0 = mean
    plt.figure(figsize=(20, 10))
    bins_values, bins, patches = plt.hist(data, label="experimental_data", density=True,
                                          bins=data.max() - data.min(),
                                          range=[data.min() - 0.5, data.max() - 0.5])
    p_experimental = bins_values


    def p_theoretical(mu):
        x = np.arange(data.min(), data.max())
        return st.poisson.pmf(x, mu)


    def l(mu):
        f = st.poisson.pmf(data, mu)
        n = f.size
        return -n * np.mean(np.log(f))

    mu_optimal = minimize(l, mu_0)['x'][0]
    plt.plot(np.arange(data.min(), data.max()), p_theoretical(mu_optimal),
             marker='o', linestyle='', color='y')
    plt.vlines(np.arange(data.min(), data.max()), ymin=0, ymax=p_theoretical(mu_optimal), colors='k', linestyles='-',
               lw=5, label=f"Poisson({np.round(mu_optimal, 3)})", color='y')
    plt.legend(fontsize=25)
    plt.xlabel('N', fontsize=25)
    plt.tick_params(labelsize=15)
    plt.savefig(f'C:/Users/user/OneDrive - Universidade do Porto/Desktop/{data.size}')

    stat, p = st.chisquare(df['N'].value_counts(normalize=True, sort=False), st.poisson.pmf(df['N'].value_counts(normalize=True, sort=False).index, mu_optimal))
    if p > 0.05:
        print(f'Chi-square goodness of fit test: statistic={np.round(stat, 2)}; p={p} > 0.05 -> Amostra segue a distribuição'
              f' de poisson (falhou a rejeitar H0)')
    else:
        print(f'Chi-square goodness of fit test: statistic={np.round(stat, 2)}; p={p} < 0.05 -> Amostra não segue a distribuição'
              f' de poisson (falhou a rejeitar H0)')

    pd.DataFrame(data=[stat, p], index=['stat', 'p'], columns=['valores']).to_excel(f'C:/Users/user/OneDrive - Universidade do Porto/Desktop/{data.size}.xlsx')
