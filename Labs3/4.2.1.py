import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as st
from statsmodels.graphics.gofplots import qqplot


# Load Data
for include in [[100], [200], [100, 200], [500], [100, 200, 700], [100, 200, 500, 700]]:
    for i, n in enumerate(include):
        df = pd.read_excel('C:/Users/user/OneDrive - Universidade do Porto/Desktop/4.2.1.xlsx', str(n))
        df['delta t (s)'] = df['t(s)'].diff().fillna(df['t(s)'].iloc[0])
        df['A (s^-1)'] = df['N'] / 0.7
        data = df['A (s^-1)'].to_numpy().flatten()
        if i == 0:
            df_ = df.copy()
        else:
            df_ = pd.concat([df_, df])
    df = df_.copy()
    data = df['A (s^-1)'].to_numpy().flatten() - 0.36

    # Calculate mean and standard deviation
    mean = np.mean(data)
    std = np.std(data, ddof=1)

    # Gaussian Tests
    data_tests = np.zeros((2, 2))
    data_tests[0] = st.normaltest(data)
    data_tests[1] = st.shapiro(data)
    df_tests = pd.DataFrame(data=data_tests, index=['D’Agostino’s K^2 Test', 'Shapiro-Wilk Test'], columns=['stat', 'p'])
    df_tests.to_excel(f'C:/Users/user/OneDrive - Universidade do Porto/Desktop/{data.size}.xlsx')

    # Histogram and qqplot of Normal distribution
    mu, sigma = st.norm.fit(data)
    x = np.linspace(data.min(), data.max(), len(data)*5)
    pdf = st.norm.pdf(x, mu, sigma)

    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(20, 10)
    fig.suptitle(f'N = {data.size}', fontsize=30)

    axes[0].hist(data, density=True, label="dados experimentais", bins=15)
    axes[0].plot(x, pdf, label=f'N({np.round(mu, 2)}; {np.round(sigma, 2)})')
    axes[0].tick_params(labelsize=15)
    axes[0].legend(fontsize=20)
    axes[0].set_xlabel(r'A ($s^{-1}$)', fontsize=20)
    axes[0].grid()

    qqplot(data, ax=axes[1], fit=True, line='s')
    axes[1].tick_params(labelsize=15)
    axes[1].set_ylabel('Sample quantiles', fontsize=20)
    axes[1].set_xlabel('Theoretical quantiles', fontsize=20)
    axes[1].grid()
    plt.savefig(f'C:/Users/user/OneDrive - Universidade do Porto/Desktop/{data.size}')